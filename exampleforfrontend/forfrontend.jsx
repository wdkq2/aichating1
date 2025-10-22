import React, { useEffect, useRef, useState } from "react";
import { ChevronLeft, Info, Home, Menu, Send, ImagePlus } from "lucide-react";

// ✅ Bugfix: 중복 import 제거 (하단에 중복 선언되어 있던 React/Lucide import 삭제)
// ▶️ 기본 아바타 (투명 1x1). 업로드 버튼으로 즉시 교체할 수 있습니다.
const PLACEHOLDER =
  "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==";

// UUID 유틸 (crypto.randomUUID 가 없을 때 대비)
const uuid = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);

// 색상 토큰: 스크린샷의 인상(화이트 바탕 + 아주 연한 회색 채팅 배경 + 짙은 블랙 텍스트 + 회색 타임스탬프)을 반영
const palette = {
  appBg: "#ffffff", // 전체 배경 (화이트)
  chatBg: "#f7f8fa", // 채팅 영역의 옅은 회색 (#F7F8FA 근사치)
  bubble: "#ffffff", // 말풍선 (화이트)
  bubbleMe: "#f9fafb", // 내 말풍선(아주 미세하게 구분)
  system: "#eef2f7", // 시스템 안내 말풍선 (한 톤 더 진한 회색)
  text: "#111827", // 본문 텍스트(거의 블랙)
  muted: "#9ca3af", // 타임스탬프/보조 텍스트
  border: "#e5e7eb", // 미세한 보더
};

function Header() {
  return (
    <header className="h-14 flex items-center justify-between border-b" style={{ borderColor: palette.border }}>
      <button className="p-2 text-gray-800" aria-label="뒤로">
        <ChevronLeft size={22} />
      </button>
      <div className="flex items-center gap-1 text-gray-900 font-semibold">
        <span className="text-base">AI톡</span>
        <Info size={16} className="opacity-80" />
      </div>
      <div className="flex items-center gap-1 text-gray-800">
        <button className="p-2" aria-label="홈">
          <Home size={20} />
        </button>
        <button className="p-2" aria-label="메뉴">
          <Menu size={20} />
        </button>
      </div>
    </header>
  );
}

function SystemPill({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="mx-auto text-xs px-4 py-2 rounded-full w-max text-gray-700"
      style={{ backgroundColor: palette.system }}
    >
      {children}
    </div>
  );
}

function AvatarCircle({ src }: { src: string }) {
  return (
    <div
      className="w-12 h-12 rounded-full overflow-hidden flex items-center justify-center ring-1"
      style={{ background: "#d1d5db", borderColor: palette.border }}
    >
      {/* 동그라미 안에 꽉 차게 */}
      <img src={src || PLACEHOLDER} alt="avatar" className="w-full h-full object-cover" />
    </div>
  );
}

function BotBubble({ text, avatar }: { text: string; avatar: string }) {
  return (
    <div className="flex items-start gap-3 mt-5">
      <AvatarCircle src={avatar} />
      <div className="max-w-[76%]">
        <div
          className="px-4 py-3 rounded-2xl shadow-sm text-[17px] leading-7 border"
          style={{ background: palette.bubble, color: palette.text, borderColor: palette.border }}
        >
          {text}
        </div>
      </div>
    </div>
  );
}

function MeBubble({ text, avatar }: { text: string; avatar: string }) {
  return (
    <div className="flex items-start gap-3 mt-4 justify-end">
      <div className="max-w-[76%]">
        <div
          className="px-4 py-3 rounded-2xl shadow-sm text-[17px] leading-7 border"
          style={{ background: palette.bubbleMe, color: palette.text, borderColor: palette.border }}
        >
          {text}
        </div>
      </div>
      <AvatarCircle src={avatar} />
    </div>
  );
}

// 유틸: 한국어 타임스탬프 포맷
function formatKRTimestamp(d: Date) {
  const h = d.getHours();
  const m = String(d.getMinutes()).padStart(2, "0");
  const s = String(d.getSeconds()).padStart(2, "0");
  const noon = h >= 12 ? "오후" : "오전";
  const hour12 = ((h + 11) % 12) + 1;
  return `${noon} ${hour12}:${m}:${s}`;
}

export default function AiTalkMock() {
  type Msg = { id: string; who: "bot" | "me"; text: string; at: Date };

  const [botAvatar, setBotAvatar] = useState<string>(PLACEHOLDER);
  const [meAvatar, setMeAvatar] = useState<string>(PLACEHOLDER);

  // 초기 메시지
  const [messages, setMessages] = useState<Msg[]>([
    {
      id: "m0",
      who: "bot",
      text: "농협은행 AI가 알아서 더치페이 결제를 탐지해요! 그리고 소비와 지출 금액을 더 정확히 기록할게요. 농협은행 소비/지출 금액을 그대로 가계부로 사용해보세요!",
      at: new Date(),
    },
  ]);

  const [input, setInput] = useState("");
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const botFileRef = useRef<HTMLInputElement | null>(null);
  const meFileRef = useRef<HTMLInputElement | null>(null);

  // 스크롤 하단 고정
  useEffect(() => {
    const el = scrollerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages.length]);

  // 👉 DEV TESTS (콘솔에서 동작). UI에는 영향 없음.
  useEffect(() => {
    const t0 = new Date(2020, 0, 1, 0, 5, 9);
    console.assert(
      formatKRTimestamp(t0).startsWith("오전 12:05:09"),
      "formatKRTimestamp: midnight case failed"
    );
    const t1 = new Date(2020, 0, 1, 12, 30, 0);
    console.assert(
      formatKRTimestamp(t1).startsWith("오후 12:30:00"),
      "formatKRTimestamp: noon case failed"
    );
    // 추가 경계 테스트
    const am = new Date(2020, 0, 1, 1, 0, 0);
    const pm = new Date(2020, 0, 1, 23, 59, 59);
    console.assert(/오전\s1:00:00/.test(formatKRTimestamp(am)), "AM boundary failed");
    console.assert(/오후\s11:59:59/.test(formatKRTimestamp(pm)), "PM boundary failed");

    console.assert(typeof uuid() === "string" && uuid().length > 0, "uuid() should return a string");
  }, []);

  // 파일 → DataURL 로드
  function readAsDataURL(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => resolve(String(fr.result));
      fr.onerror = reject;
      fr.readAsDataURL(file);
    });
  }

  async function onPickAvatar(e: React.ChangeEvent<HTMLInputElement>, target: "bot" | "me") {
    const file = e.target.files?.[0];
    if (!file) return;
    const data = await readAsDataURL(file);
    if (target === "bot") setBotAvatar(data);
    else setMeAvatar(data);
    // 입력 초기화 (같은 파일 재선택 시 change 이벤트 보장)
    e.target.value = "";
  }

  function sendMessage() {
    const text = input.trim();
    if (!text) return;
    const now = new Date();
    const myMsg: Msg = { id: uuid(), who: "me", text, at: now };
    setMessages((prev) => [...prev, myMsg]);
    setInput("");

    // 데모용 간단한 응답(에코 + 안내)
    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        {
          id: uuid(),
          who: "bot",
          text: `“${text}”에 대한 정보를 정리해 드릴게요!`,
          at: new Date(),
        },
      ]);
    }, 400);
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="min-h-screen w-full flex items-center justify-center p-6" style={{ background: palette.appBg }}>
      {/* 모바일 프레임 */}
      <div
        className="w-[390px] h-[700px] md:h-[760px] rounded-3xl overflow-hidden shadow-xl flex flex-col relative"
        style={{ border: `1px solid ${palette.border}`, background: "#fff" }}
      >
        <Header />

        {/* 스크롤 영역 */}
        <div className="flex-1" style={{ background: palette.chatBg }}>
          <div ref={scrollerRef} className="h-full overflow-y-auto px-4 pt-4 pb-32">
            <SystemPill>AI톡과 대화를 시작합니다.</SystemPill>

            {messages.map((m) =>
              m.who === "bot" ? (
                <div key={m.id}>
                  <BotBubble text={m.text} avatar={botAvatar} />
                  <div className="mt-2 pr-1 text-right text-[12px]" style={{ color: palette.muted }}>
                    {formatKRTimestamp(m.at)}
                  </div>
                </div>
              ) : (
                <div key={m.id}>
                  <MeBubble text={m.text} avatar={meAvatar} />
                  <div className="mt-2 pl-1 text-left text-[12px]" style={{ color: palette.muted }}>
                    {formatKRTimestamp(m.at)}
                  </div>
                </div>
              )
            )}
          </div>
        </div>

        {/* 입력 영역 */}
        <div className="absolute left-0 right-0 bottom-0">
          <div className="px-3 pb-3" style={{ background: palette.chatBg }}>
            {/* 아바타 업로드 단추들 */}
            <div className="flex justify-between items-center mb-2 px-1">
              <div className="flex items-center gap-3 text-[12px]" style={{ color: palette.muted }}>
                <button
                  className="flex items-center gap-1 px-2 py-1 rounded-md border"
                  style={{ borderColor: palette.border, background: "#fff" }}
                  onClick={() => botFileRef.current?.click()}
                  title="상대방 아바타 변경"
                >
                  <ImagePlus size={16} /> 상대 아바타
                </button>
                <button
                  className="flex items-center gap-1 px-2 py-1 rounded-md border"
                  style={{ borderColor: palette.border, background: "#fff" }}
                  onClick={() => meFileRef.current?.click()}
                  title="내 아바타 변경"
                >
                  <ImagePlus size={16} /> 내 아바타
                </button>
              </div>
            </div>

            <div className="flex items-center gap-2 p-2 rounded-2xl border bg-white" style={{ borderColor: palette.border }}>
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder="메시지를 입력하세요 (Enter 전송, Shift+Enter 줄바꿈)"
                className="flex-1 outline-none text-[16px] px-2 bg-transparent"
              />
              <button
                onClick={sendMessage}
                className="p-2 rounded-xl border disabled:opacity-50"
                style={{ borderColor: palette.border, background: "#fff" }}
                disabled={!input.trim()}
                aria-label="전송"
              >
                <Send size={18} />
              </button>
            </div>
          </div>
        </div>

        {/* 숨겨진 파일 인풋: bot / me 아바타 */}
        <input ref={botFileRef} type="file" accept="image/*" className="hidden" onChange={(e) => onPickAvatar(e, "bot")} />
        <input ref={meFileRef} type="file" accept="image/*" className="hidden" onChange={(e) => onPickAvatar(e, "me")} />
      </div>
    </div>
  );
}
