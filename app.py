#!/usr/bin/env python3
"""
A lightweight demonstration backend implementing Dutch pay detection (v3.1) with
persona-aware share estimation and simple HTTP API.  This server supports two
endpoints for posting payment and deposit events and returns chat-like
notifications when a candidate dutch pay is detected or when a settlement
completes.  It also serves a small static frontend for interactive demos.

The detection logic follows these rules:
1. Only expense transactions with `category` of ``restaurants`` or ``cafe``
   are considered for dutch‑pay.  Other categories are ignored.
2. A transaction is considered a "large" expense if it exceeds a threshold
   based on robust statistics (median and IQR) across the persona's past
   expenses.  This threshold adapts to the time of day and weekday/weekend.
3. When a large expense is found, the engine estimates the expected party
   size and per‑person share using both the learned statistics and persona
   seed values.  Expected deposits are constrained to amounts close to
   one share (±30%).  The required number of deposits grows with the party
   size (n̂ − 1).
4. Deposits within a configurable time window (default 48 hours) after the
   expense are matched to the candidate.  When enough deposits arrive and
   their total ratio falls within the expected range, the candidate is
   confirmed and a settlement notification is emitted.

This module intentionally avoids external dependencies (no Flask) and uses
Python's built‑in ``http.server``.  It is suitable for demonstration
purposes but not production use.
"""

import json
import os
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from datetime import datetime, timedelta, timezone

######################################################################
# Persona definitions
#
# Each persona describes typical per‑person spend for the time bucket,
# category and weekday/weekend.  The ``share_seed`` table now fully defines the
# expected per-person share; ``mix_k`` is retained for backward compatibility
# but new observations no longer alter these seed values.
######################################################################
PERSONAS: dict = {
    "P1": {
        "id": "P1",
        "label": "페르소나 1 (20대 남성)",
        "notes": "저녁 외식 비중↑, 카페 소액 빈도 중간",
        "mix_k": 5,
        "share_seed": {
            "restaurants": {
                "weekday": {
                    "morning": 7000, "lunch": 9000, "afternoon": 8000, "dinner": 12000, "latenight": 10000
                },
                "weekend": {
                    "morning": 8000, "lunch": 10000, "afternoon": 9000, "dinner": 14000, "latenight": 11000
                }
            },
            "cafe": {
                "weekday": {
                    "morning": 5000, "lunch": 5500, "afternoon": 6000, "dinner": 5500, "latenight": 0
                },
                "weekend": {
                    "morning": 5500, "lunch": 6000, "afternoon": 6500, "dinner": 6000, "latenight": 0
                }
            }
        }
    },
    "P2": {
        "id": "P2",
        "label": "페르소나 2 (40대 여성)",
        "notes": "점심·주말 외식 단가↑, 카페 이용 잦음",
        "mix_k": 5,
        "share_seed": {
            "restaurants": {
                "weekday": {
                    "morning": 9000, "lunch": 13000, "afternoon": 11000, "dinner": 18000, "latenight": 12000
                },
                "weekend": {
                    "morning": 10000, "lunch": 15000, "afternoon": 12000, "dinner": 20000, "latenight": 13000
                }
            },
            "cafe": {
                "weekday": {
                    "morning": 6000, "lunch": 6500, "afternoon": 7000, "dinner": 6500, "latenight": 0
                },
                "weekend": {
                    "morning": 6500, "lunch": 7000, "afternoon": 7500, "dinner": 7000, "latenight": 0
                }
            }
        }
    },
    "P3": {
        "id": "P3",
        "label": "페르소나 3 (50대 남성)",
        "notes": "저녁 외식 단가↑, 카페 빈도 낮음",
        "mix_k": 5,
        "share_seed": {
            "restaurants": {
                "weekday": {
                    "morning": 9000, "lunch": 15000, "afternoon": 12000, "dinner": 20000, "latenight": 14000
                },
                "weekend": {
                    "morning": 10000, "lunch": 16000, "afternoon": 13000, "dinner": 22000, "latenight": 15000
                }
            },
            "cafe": {
                "weekday": {
                    "morning": 4500, "lunch": 5000, "afternoon": 5500, "dinner": 5000, "latenight": 0
                },
                "weekend": {
                    "morning": 5000, "lunch": 5500, "afternoon": 6000, "dinner": 5500, "latenight": 0
                }
            }
        }
    }
}

# Global event store.  Each persona id maps to a list of events where each
# event has an incremental event_id and the event payload as returned by the
# detection engine.  This enables the chat frontend to poll new messages.
EVENT_LOG: dict = {}

def record_event(persona_id: str, event: dict):
    """
    Record a chat event for the given persona.  Each event is stored with
    a monotonically increasing integer id so clients can request events
    after a known point.  If no prior events exist for the persona, the
    log is initialised.

    :param persona_id: persona identifier
    :param event: event dict produced by engine.process_payment or
                  engine.process_deposit
    """
    log = EVENT_LOG.setdefault(persona_id, [])
    next_id = log[-1]["event_id"] + 1 if log else 1
    log.append({"event_id": next_id, "event": event})


######################################################################
# Detection Engine
######################################################################

class DutchPayEngine:
    """
    Per‑persona dutch pay detection engine.  Maintains the transaction
    history for a single persona and tracks active dutch‑pay candidates.
    """

    # Detection hyperparameters
    ALLOWED_CATEGORIES = {"restaurants", "cafe"}
    WINDOW_HOURS = 48
    SHARE_TOL_LOW = 0.7
    SHARE_TOL_HIGH = 1.3
    RATIO_TOLERANCE = 0.15
    CV_MAX = 0.8
    PARTY_MAX = 10
    # Large expense threshold parameters
    LARGE_FACTOR = 2.0
    IQR_MULT = 1.0
    MIN_EXPENSE_AMOUNT = 30000

    def __init__(self, persona: dict):
        self.persona = persona
        # Number of observations required to fully override seed.  See
        # ``compute_share``.
        self.mix_k: float = persona.get("mix_k", 5)
        self.share_seed: dict = persona.get("share_seed", {})
        # All transactions for this persona (expenses and deposits).  Each
        # transaction is a dict with id, datetime (as datetime), amount (float),
        # type ("expense" or "deposit"), category (if expense), etc.
        self.transactions: list = []
        # Active dutch‑pay candidate state keyed by payment id
        self.candidates: dict = {}
        # Set of deposit ids that have been consumed
        self.used_deposits: set = set()

    # Helper: parse input date/time strings to datetime.  Accepts ISO‑8601
    # (with or without timezone) and ``YYYY-MM-DDTHH:MM`` formats.  Falls back to
    # naive parsing if timezone missing.  Do not adjust timezone.
    def _to_dt(self, x) -> datetime:
        if isinstance(x, datetime):
            return x
        s = str(x)
        try:
            # Python 3.8+ can parse ISO with timezone
            return datetime.fromisoformat(s)
        except Exception:
            try:
                return datetime.strptime(s, "%Y-%m-%dT%H:%M")
            except Exception:
                try:
                    return datetime.strptime(s, "%Y-%m-%d %H:%M")
                except Exception:
                    raise ValueError(f"Unrecognised datetime format: {x}")

    # Determine the bucket for a datetime (morning, lunch, afternoon, dinner,
    # latenight or other).  Note that latenight covers [22,23] and [0,2].
    @staticmethod 		
    def _bucket_of(dt: datetime) -> str:
        h = dt.hour
        if h >= 22 or h <= 2:
            return "latenight"
        if 6 <= h <= 10:
            return "morning"
        if 11 <= h <= 14:
            return "lunch"
        if 15 <= h <= 17:
            return "afternoon"
        if 18 <= h <= 21:
            return "dinner"
        return "other"

    # Determine weekday/weekend label
    @staticmethod
    def _weektag(dt: datetime) -> str:
        return "weekend" if dt.weekday() >= 5 else "weekday"

    # Compute percentile of sorted values
    @staticmethod
    def _percentile(vs, p: float) -> float:
        n = len(vs)
        if n == 0:
            return 0.0
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return vs[f]
        return vs[f] + (vs[c] - vs[f]) * (k - f)

    # Compute baseline statistics from existing expense transactions.  Returns
    # ``{
    #     'global': {'median': float, 'q1': float, 'q3': float, 'iqr': float, 'count': int},
    #     'ctx': {<context_key>: {'median': float, 'q3': float, 'iqr': float, 'count': int}}
    # }``
    def _compute_baseline(self):
        # Gather all expense amounts (positive only)
        amounts = []
        ctx_vals = {}
        _ = max([tx["datetime"] for tx in self.transactions] or [datetime.now(timezone.utc)])
        # Gather known candidate payment ids so we can exclude them from the
        # baseline even if they predate the "exclude_from_baseline" flag (for
        # example transactions recorded before the v3.1.1 patch).  This keeps
        # long-running servers consistent after upgrades.
        excluded_ids = {
            pid
            for cand in self.candidates.values()
            for pid in [cand.get("payment", {}).get("id")]
            if pid and cand.get("state") in {"CANDIDATE", "CONFIRMED"}
        }
        for tx in self.transactions:
            if tx.get("type") != "deposit" and tx.get("amount", 0) > 0:
                # Exclude transactions that have been marked as dutch-pay
                # candidates/confirmed settlements from the baseline so that
                # exceptionally large group payments do not inflate the
                # personal spending profile.  Without this guard, a confirmed
                # dutch-pay would double the effective median, preventing the
                # next legitimate candidate from being detected.
                if tx.get("exclude_from_baseline") or tx.get("id") in excluded_ids:
                    continue
                # We do not apply a lookback cutoff here; the engine may be
                # extended with lookback in future
                amt = float(tx["amount"])
                amounts.append(amt)
                # context key for robust statistics
                key = f"{self._bucket_of(tx['datetime'])}:{self._weektag(tx['datetime'])}"
                ctx_vals.setdefault(key, []).append(amt)
        if not amounts:
            # Provide a reasonable default baseline if no history exists
            default_median = 15000.0
            return {
                "global": {"median": default_median, "q1": default_median*0.67, "q3": default_median*1.33,
                            "iqr": default_median*0.66, "count": 0},
                "ctx": {}
            }
        amounts_sorted = sorted(amounts)
        global_median = self._percentile(amounts_sorted, 0.5)
        q1 = self._percentile(amounts_sorted, 0.25)
        q3 = self._percentile(amounts_sorted, 0.75)
        iqr = max(1000.0, q3 - q1)
        baseline = {
            "global": {
                "median": global_median,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "count": len(amounts_sorted)
            },
            "ctx": {}
        }
        # Compute context‑specific stats
        for key, vals in ctx_vals.items():
            vs = sorted(vals)
            md = self._percentile(vs, 0.5)
            q3c = self._percentile(vs, 0.75)
            iqr_c = max(1000.0, q3c - self._percentile(vs, 0.25))
            baseline["ctx"][key] = {
                "median": md,
                "q3": q3c,
                "iqr": iqr_c,
                "count": len(vs)
            }
        return baseline

    # Blend context stats with global using a simple weighted average based on
    # alpha prior (similar to v2).  Not used for share but used for large
    # expense threshold.
    def _blend(self, g: float, c: float, n_ctx: int, alpha: float) -> float:
        return (n_ctx * c + alpha * g) / max(1.0, n_ctx + alpha)

    # Determine if an expense is large.  Returns (bool, debug_info)
    def _is_large_expense(self, amount: float, dt: datetime, baseline):
        g = baseline["global"]
        ctx_key = f"{self._bucket_of(dt)}:{self._weektag(dt)}"
        ctx = baseline["ctx"].get(ctx_key, {"median": g["median"], "q3": g["q3"], "iqr": g["iqr"], "count": 0})
        # Blend median, q3, iqr with alpha prior for stability
        alpha = 8.0
        md = self._blend(g["median"], ctx["median"], ctx["count"], alpha)
        q3 = self._blend(g["q3"], ctx["q3"], ctx["count"], alpha)
        iqr = self._blend(g["iqr"], ctx["iqr"], ctx["count"], alpha)
        # Two candidate thresholds: median * factor vs q3 + IQR * mult
        rel_thr = self.LARGE_FACTOR * md
        iqr_thr = q3 + self.IQR_MULT * iqr
        thr = max(rel_thr, iqr_thr)
        abs_ok = amount >= self.MIN_EXPENSE_AMOUNT
        is_large = abs_ok and (amount >= thr)
        debug = {
            "median": md,
            "q3": q3,
            "iqr": iqr,
            "threshold": thr,
            "rel_thr": rel_thr,
            "iqr_thr": iqr_thr,
            "ctx_key": ctx_key,
            "ctx_count": ctx.get("count", 0)
        }
        return is_large, debug

    # Compute the expected per‑person share for a given time context and category.
    # The value is derived exclusively from the persona's seed table so it remains
    # stable regardless of newly observed transactions.
    def _compute_share(self, dt: datetime, category: str, baseline) -> float:
        """Return the per-person share using only persona seed values.

        Operational feedback should not modify the expected share; instead we
        rely solely on the fixed seed tables defined for each persona.  The
        ``baseline`` argument is still accepted for signature compatibility but
        is no longer consulted when computing the share.
        """

        def _seed_lookup(cat_seed: dict, week: str, bucket: str):
            val = cat_seed.get(week, {}).get(bucket)
            return float(val) if val else None

        def _seed_fallback(cat_seed: dict, week: str):
            # Prefer other buckets in the same week, then all remaining seeds.
            week_vals = [
                float(v)
                for v in cat_seed.get(week, {}).values()
                if v and float(v) > 0
            ]
            if week_vals:
                return sum(week_vals) / len(week_vals)
            all_vals = [
                float(v)
                for week_map in cat_seed.values()
                for v in week_map.values()
                if v and float(v) > 0
            ]
            if all_vals:
                return sum(all_vals) / len(all_vals)
            # Final fallback: reasonable constant so detection can proceed.
            return 15000.0

        cat_seed = self.share_seed.get(category, {})
        persona_week = self._weektag(dt)
        persona_bucket = self._bucket_of(dt)
        share = _seed_lookup(cat_seed, persona_week, persona_bucket)
        if share is None:
            share = _seed_fallback(cat_seed, persona_week)
        return max(1.0, float(share))

    # Estimate number of people sharing and per‑person share
    def _estimate_party(self, amount: float, dt: datetime, category: str, baseline):
        s_hat = self._compute_share(dt, category, baseline)
        # Avoid division by zero
        if s_hat <= 0:
            s_hat = 1.0
        n_hat = max(2, min(self.PARTY_MAX, int(round(amount / s_hat))))
        s_hat = amount / n_hat if n_hat > 0 else amount
        return n_hat, s_hat

    # Compute CV of a list of values
    @staticmethod
    def _coeff_var(vals):
        n = len(vals)
        if n == 0:
            return 0.0
        mean = sum(vals) / n
        if mean == 0:
            return 0.0
        var = sum((x - mean) ** 2 for x in vals) / n
        return math.sqrt(var) / mean

    def process_payment(self, tx: dict):
        """
        Process a single payment (expense) event.  If the payment qualifies as
        a dutch‑pay candidate, a candidate prompt dictionary is returned.  The
        returned dict is suitable for sending to the frontend chat.  If the
        payment does not trigger a candidate, returns None.
        """
        # Normalise transaction fields
        tx = dict(tx)
        tx_dt = self._to_dt(tx.get("datetime"))
        tx_amount = float(tx.get("amount", 0))
        tx_id = tx.get("id") or f"tx_{len(self.transactions)}"
        tx_category = str(tx.get("category", "")).lower()
        # Only consider allowed categories
        if tx_category not in self.ALLOWED_CATEGORIES:
            # Append to history for completeness but do not consider for dutch pay
            tx_obj = {
                "id": tx_id,
                "datetime": tx_dt,
                "amount": tx_amount,
                "type": "expense",
                "category": tx_category
            }
            self.transactions.append(tx_obj)
            return None
        # Compute baseline BEFORE adding this expense to history
        baseline = self._compute_baseline()
        # Test if the payment is large
        is_large, dbg = self._is_large_expense(tx_amount, tx_dt, baseline)
        # Now append transaction to history (always)
        tx_obj = {
            "id": tx_id,
            "datetime": tx_dt,
            "amount": tx_amount,
            "type": "expense",
            "category": tx_category,
            "baseline_debug": dbg
        }
        self.transactions.append(tx_obj)
        if not is_large:
            return None
        # Mark large expenses so they are ignored when building future
        # baselines.  This keeps confirmed dutch-pay expenses from raising the
        # thresholds and suppressing subsequent detections for the same
        # persona.
        tx_obj["exclude_from_baseline"] = True
        # Estimate party size and per‑person share using baseline
        n_hat, s_hat = self._estimate_party(tx_amount, tx_dt, tx_category, baseline)
        # Minimum deposits required (n_hat - 1), bounded
        min_reimb = max(1, min(n_hat - 1, self.PARTY_MAX))
        rho_star = (n_hat - 1) / n_hat
        lower_ratio = max(0.0, rho_star - self.RATIO_TOLERANCE)
        upper_ratio = rho_star + self.RATIO_TOLERANCE
        cand = {
            "payment": tx_obj,
            "state": "CANDIDATE",
            "deposits": [],
            "sum_deposits": 0.0,
            "n_hat": n_hat,
            "s_hat": s_hat,
            "min_reimb": min_reimb,
            "ratio_bounds": (lower_ratio, upper_ratio)
        }
        self.candidates[tx_id] = cand
        prompt = {
            "type": "CANDIDATE_PROMPT",
            "payment_id": tx_id,
            "payload": {
                "title": "방금 결제하신 것은 더치페이인가요?",
                "category": tx_category,
                "amount_krw": int(tx_amount),
                "datetime": tx_dt.strftime("%Y-%m-%d %H:%M (KST)"),
                "actions": [
                    {"label": "네", "action": "LABEL_YES", "payment_id": tx_id},
                    {"label": "아니오", "action": "LABEL_NO", "payment_id": tx_id}
                ]
            }
        }
        return prompt

    def process_deposit(self, tx: dict):
        """
        Process a deposit event.  Attempts to match the deposit to the most
        recent candidate in the time window that still requires reimbursement.
        Returns a settlement notification dict when a candidate is confirmed.
        Otherwise returns None.
        """
        tx = dict(tx)
        tx["datetime"] = self._to_dt(tx.get("datetime"))
        tx["amount"] = float(tx.get("amount", 0))
        tx.setdefault("id", f"dp_{len(self.transactions)}")
        tx["type"] = "deposit"
        # Append to history (does not affect baseline computations for expenses)
        self.transactions.append(tx)
        # Skip if deposit already used
        if tx["id"] in self.used_deposits:
            return None
        # Sort candidates by payment time descending (most recent first)
        candidates_sorted = sorted(
            [c for c in self.candidates.values() if c["state"] == "CANDIDATE"],
            key=lambda c: c["payment"]["datetime"], reverse=True
        )
        for cand in candidates_sorted:
            pay_tx = cand["payment"]
            # Check time window
            start = pay_tx["datetime"]
            end = start + timedelta(hours=self.WINDOW_HOURS)
            if not (tx["datetime"] >= start and tx["datetime"] <= end):
                continue
            # Deposit must be within share tolerance
            s_hat = cand["s_hat"]
            v = tx["amount"]
            low = s_hat * self.SHARE_TOL_LOW
            high = s_hat * self.SHARE_TOL_HIGH
            if v < low or v > high:
                # Try to infer a smaller party size when deposits are larger
                # than expected.  This happens when the original share
                # estimate (based purely on historical medians) is too low for
                # the newly observed group size.  Use the deposit amount to
                # re-estimate the per-person share and relax the minimum deposit
                # requirement so settlements can complete.
                inferred_n = int(round(pay_tx["amount"] / max(v, 1.0)))
                inferred_n = max(2, min(self.PARTY_MAX, inferred_n))
                inferred_share = pay_tx["amount"] / inferred_n if inferred_n else s_hat
                inferred_low = inferred_share * self.SHARE_TOL_LOW
                inferred_high = inferred_share * self.SHARE_TOL_HIGH
                if v < inferred_low or v > inferred_high:
                    continue
                # Accept the deposit and update candidate expectations.
                s_hat = inferred_share
                cand["s_hat"] = s_hat
                cand["n_hat"] = inferred_n
                cand["min_reimb"] = max(1, min(inferred_n - 1, self.PARTY_MAX))
                rho_star = (inferred_n - 1) / inferred_n
                cand["ratio_bounds"] = (
                    max(0.0, rho_star - self.RATIO_TOLERANCE),
                    rho_star + self.RATIO_TOLERANCE
                )
            # Accept deposit for this candidate
            cand["deposits"].append(tx)
            cand["sum_deposits"] += v
            self.used_deposits.add(tx["id"])
            # Evaluate if candidate is now confirmed
            cnt = len(cand["deposits"])
            pay_amount = max(1.0, pay_tx["amount"])
            # Re-estimate party size when the observed deposits imply a
            # smaller group than originally predicted.  This helps scenarios
            # where the seed-based share slightly underestimates the actual
            # per-person amount (e.g. lunch vs. dinner), preventing the
            # required deposit count and ratio bounds from remaining too
            # strict.
            avg_share = cand["sum_deposits"] / cnt if cnt else cand["s_hat"]
            if avg_share > 0:
                implied_n = int(round(pay_amount / avg_share))
                implied_n = max(2, min(self.PARTY_MAX, implied_n))
                if implied_n < cand["n_hat"]:
                    cand["n_hat"] = implied_n
                    cand["s_hat"] = pay_amount / implied_n
                    cand["min_reimb"] = max(1, min(implied_n - 1, self.PARTY_MAX))
                    rho_star = (implied_n - 1) / implied_n
                    cand["ratio_bounds"] = (
                        max(0.0, rho_star - self.RATIO_TOLERANCE),
                        rho_star + self.RATIO_TOLERANCE
                    )
            # Compute current ratio of reimbursements
            ratio = cand["sum_deposits"] / pay_amount
            lb, ub = cand["ratio_bounds"]
            # Check minimum count and ratio window
            if cnt >= cand["min_reimb"] and lb <= ratio <= ub:
                # For parties larger than 2 people, ensure coefficient of variation constraint
                if cnt > 1:
                    amounts = [d["amount"] for d in cand["deposits"]]
                    cv = self._coeff_var(amounts)
                    if cv > self.CV_MAX:
                        # Do not confirm if variation too high
                        break
                # Candidate confirmed
                cand["state"] = "CONFIRMED"
                # Ensure the payment never contributes to future baselines
                pay_tx.setdefault("exclude_from_baseline", True)
                adjusted_amount = max(0.0, pay_tx["amount"] - cand["sum_deposits"])
                # Build settlement notification
                settlement = {
                    "type": "SETTLEMENT_COMPLETED",
                    "payment_id": pay_tx["id"],
                    "payload": {
                        "title": "더치페이 정산이 완료되었습니다!",
                        "category": str(pay_tx.get("category", "")).lower(),
                        "amount_krw": int(pay_tx["amount"]),
                        "datetime": pay_tx["datetime"].strftime("%Y-%m-%d %H:%M (KST)"),
                        "party_size_estimated": cand["n_hat"],
                        "expected_share_estimated": int(round(cand["s_hat"])),
                        "deposits_count": cnt,
                        "deposits_total": int(round(cand["sum_deposits"])),
                        "adjusted_amount": int(round(adjusted_amount))
                    }
                }
                return settlement
            # If not confirmed yet, continue searching (deposit cannot match another candidate)
            return None
        # Deposit not matched to any candidate
        return None

    def label_payment(self, payment_id: str, label: str):
        """Apply a manual label to a candidate payment.

        Returning ``True`` indicates that the payment existed.  A ``NO`` label
        dismisses the candidate entirely so that future deposits no longer try
        to match against it and any already-consumed reimbursements become
        available for other candidates.  ``YES`` currently only records the
        feedback for telemetry purposes.
        """

        cand = self.candidates.get(payment_id)
        if not cand:
            return False

        cand.setdefault("user_label", label)

        if label == "NO":
            # Release any deposits that may have been tentatively attributed to
            # this candidate so they remain usable if another dutch-pay is
            # active in the same window.
            for dep in cand.get("deposits", []):
                dep_id = dep.get("id")
                if dep_id:
                    self.used_deposits.discard(dep_id)
            pay_tx = cand.get("payment") or {}
            # Remove the expense from the transaction history entirely so it
            # does not influence future baselines or party-size estimation.
            pay_id = pay_tx.get("id")
            if pay_id:
                self.transactions = [
                    tx for tx in self.transactions if tx.get("id") != pay_id
                ]
            cand["state"] = "DISMISSED"
            # Remove the candidate entirely so it cannot interfere with other
            # payments.
            self.candidates.pop(payment_id, None)
        return True

######################################################################
# HTTP Handler
######################################################################

class DutchPayHTTPRequestHandler(BaseHTTPRequestHandler):
    """
    Minimal HTTP handler to serve the static frontend and expose the dutch pay
    API.  It supports:
      - GET /personas : returns available personas as JSON
      - POST /payments : accepts a single payment (expense) JSON and
                         returns a candidate prompt or nothing
      - POST /deposits : accepts a single deposit JSON and returns a
                         settlement notification or nothing
      - POST /payments/<payment_id>/label : records a user label (YES/NO)
      - Static file serving from ``static`` directory
    """

    # Map persona_id to engine
    engines = {pid: DutchPayEngine(persona) for pid, persona in PERSONAS.items()}

    def _send_json(self, code: int, data):
        body = json.dumps(data or {}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        # Root serves input page
        if path == "/":
            return self._serve_file("static/input.html")
        # Personas
        if path == "/personas":
            # Provide a list of personas (id, label, notes)
            personas_list = [
                {"id": p["id"], "label": p["label"], "notes": p.get("notes", "")}
                for p in PERSONAS.values()
            ]
            return self._send_json(200, {"personas": personas_list})
        # Events polling.  Clients may request new chat events by persona and optional
        # after id (only events with id greater than this value are returned).
        if path == "/events":
            # Query string example: /events?persona_id=P1&after=3
            query = parsed.query
            params = {}
            for kv in query.split('&'):
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    params[k] = v
            persona_id = params.get('persona_id')
            if not persona_id or persona_id not in PERSONAS:
                return self._send_json(400, {"error": "invalid persona_id"})
            after = int(params.get('after') or 0)
            log = EVENT_LOG.get(persona_id, [])
            events = [entry for entry in log if entry["event_id"] > after]
            # Return only the event objects (without internal event_id) but include event_id
            resp = [ {"event_id": e["event_id"], **e["event"]} for e in events ]
            return self._send_json(200, {"events": resp})
        # Static file
        if path.startswith("/static/"):
            return self._serve_file(path.lstrip("/"))
        # Not found
        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        content_length = int(self.headers.get('Content-Length', '0'))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        try:
            data = json.loads(body.decode("utf-8") or "{}")
        except Exception:
            data = {}
        # POST /payments
        if path == "/payments":
            persona_id = data.get("persona_id")
            if not persona_id or persona_id not in self.engines:
                return self._send_json(400, {"error": "invalid persona_id"})
            engine = self.engines[persona_id]
            prompt = engine.process_payment(data)
            if prompt:
                # Record event so chat clients can pick it up
                record_event(persona_id, prompt)
                return self._send_json(200, prompt)
            return self._send_json(200, {})
        # POST /deposits
        if path == "/deposits":
            persona_id = data.get("persona_id")
            if not persona_id or persona_id not in self.engines:
                return self._send_json(400, {"error": "invalid persona_id"})
            engine = self.engines[persona_id]
            settlement = engine.process_deposit(data)
            if settlement:
                # Record event for settlement completion
                record_event(persona_id, settlement)
                return self._send_json(200, settlement)
            return self._send_json(200, {})
        # POST /payments/<id>/label
        if path.startswith("/payments/") and path.endswith("/label"):
            parts = path.split("/")
            if len(parts) >= 4:
                payment_id = parts[2]
                # Determine persona engine containing this payment
                label = str(data.get("label", "")).upper()
                found = False
                for engine in self.engines.values():
                    if engine.label_payment(payment_id, label):
                        found = True
                        break
                if found:
                    return self._send_json(200, {"status": "ok"})
            return self._send_json(404, {"error": "payment not found"})
        # Unknown endpoint
        self.send_error(404, "Not Found")

    def _serve_file(self, relative_path: str):
        # Avoid directory traversal
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        path = os.path.normpath(os.path.join(static_dir, os.path.relpath(relative_path, 'static')))
        if not path.startswith(static_dir):
            self.send_error(403, "Forbidden")
            return
        if not os.path.exists(path) or not os.path.isfile(path):
            self.send_error(404, "Not Found")
            return
        try:
            with open(path, 'rb') as f:
                data = f.read()
            ext = os.path.splitext(path)[1].lower()
            if ext == '.html':
                ctype = 'text/html; charset=utf-8'
            elif ext == '.js':
                ctype = 'application/javascript; charset=utf-8'
            elif ext == '.css':
                ctype = 'text/css; charset=utf-8'
            elif ext in ('.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp'):
                ctype = 'image/' + ext.lstrip('.')
            else:
                ctype = 'application/octet-stream'
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")


def run_server(port: int = 8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, DutchPayHTTPRequestHandler)
    print(f"Running dutchpay demo server on http://localhost:{port} ...")
    httpd.serve_forever()


if __name__ == '__main__':
    port_env = os.environ.get('PORT')
    try:
        port = int(port_env) if port_env else 8000
    except ValueError:
        port = 8000
    run_server(port)