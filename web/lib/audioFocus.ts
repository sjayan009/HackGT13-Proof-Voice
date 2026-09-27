// One audible source at a time: any player that starts announces itself and every other player pauses.

const EVENT = "proofvoice:audio-focus";

export function claimAudioFocus(owner: string) {
  window.dispatchEvent(new CustomEvent(EVENT, { detail: owner }));
}

export function onAudioFocus(owner: string, pause: () => void): () => void {
  const handler = (e: Event) => {
    if ((e as CustomEvent<string>).detail !== owner) pause();
  };
  window.addEventListener(EVENT, handler);
  return () => window.removeEventListener(EVENT, handler);
}
