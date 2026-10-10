import React, { useEffect, useState } from "react";

const reducedMotion = () =>
  typeof window !== "undefined" &&
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/* A heading that types itself, then blinks a terminal cursor a few times and
   removes it. Screen readers get the full text at once. */
export function TypedTitle({ text, as: Tag = "h1", speed = 45, blinks = 3 }) {
  const still = reducedMotion();
  const [shown, setShown] = useState(still ? text.length : 0),
    [cursor, setCursor] = useState(!still);
  useEffect(() => {
    if (still) return;
    if (shown < text.length) {
      const t = setTimeout(() => setShown(shown + 1), speed);
      return () => clearTimeout(t);
    }
    // each blink is one 530ms on/off cycle, as in a terminal
    const t = setTimeout(() => setCursor(false), blinks * 1060);
    return () => clearTimeout(t);
  }, [shown, text, speed, blinks, still]);
  return (
    <Tag className="typed-title" aria-label={text}>
      <span aria-hidden="true">{text.slice(0, shown)}</span>
      {cursor && (
        <span
          className={`terminal-cursor ${shown < text.length ? "is-typing" : ""}`}
          aria-hidden="true"
        />
      )}
      {/* reserves the final width so the layout does not shift while typing */}
      <span className="typed-ghost" aria-hidden="true">
        {text.slice(shown)}
      </span>
    </Tag>
  );
}

/* A number that counts up from zero when it first appears. */
export function CountUp({ value, duration = 900 }) {
  const [n, setN] = useState(reducedMotion() ? value : 0);
  useEffect(() => {
    if (reducedMotion()) return setN(value);
    let frame, start;
    const step = (now) => {
      start ??= now;
      const p = Math.min((now - start) / duration, 1);
      setN(Math.round(value * (1 - Math.pow(1 - p, 3))));
      if (p < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [value, duration]);
  return <>{n}</>;
}

export function greeting(date = new Date()) {
  const h = date.getHours();
  return h < 5
    ? "Working late"
    : h < 12
      ? "Good morning"
      : h < 18
        ? "Good afternoon"
        : "Good evening";
}
