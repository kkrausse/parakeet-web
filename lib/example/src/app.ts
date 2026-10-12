import { createDictation } from "parakeet-web";

const text = document.querySelector<HTMLTextAreaElement>("#text")!;
const button = document.querySelector<HTMLButtonElement>("#rec")!;
const status = document.querySelector<HTMLElement>("#status")!;
let committed = "";
const dictation = createDictation({
  assetsUrl: "/parakeet-web/",
  modelUrl: "/models/parakeet-tdt_ctc-110m-Q4_0.gguf",
  onCommit(words) { committed += (committed ? " " : "") + words; text.value = committed; },
  onTentative(guess) { text.value = committed + (guess ? " " + guess : ""); },
  onState(s) { button.textContent = s.recording ? "Stop" : "Record"; status.textContent = `model ${s.model}, microphone ${s.microphone}`; },
  onLoadProgress(p) { status.textContent = `loading ${Math.round(p.fraction * 100)}%`; },
  onError(e) { status.textContent = e.message; },
});
button.onclick = () => (dictation.state.recording ? dictation.stop() : dictation.start());
