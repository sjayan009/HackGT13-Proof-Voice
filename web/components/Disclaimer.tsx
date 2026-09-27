import Icon from "./Icon";

export default function Disclaimer({ text }: { text: string }) {
  if (!text) return null;
  return (
    <p className="disclaimer">
      <Icon name="info" size={14} />
      <span>{text}</span>
    </p>
  );
}
