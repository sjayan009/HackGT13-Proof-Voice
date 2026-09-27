import Icon from "./Icon";

export default function Callout({
  tone = "info",
  title,
  children,
  action,
}: {
  tone?: "info" | "error" | "warn";
  title?: string;
  children: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <div className={`callout callout-${tone} appear`} role={tone === "error" ? "alert" : "status"}>
      <Icon name={tone === "info" ? "info" : "alert"} size={16} />
      <div className="callout-body">
        {title && <span className="callout-title">{title}</span>}
        {children}
      </div>
      {action}
    </div>
  );
}
