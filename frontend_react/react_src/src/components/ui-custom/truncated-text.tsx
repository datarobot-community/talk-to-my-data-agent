import React from "react";
import { cn } from "@/lib/utils";

interface TruncatedTextProps {
  text?: string;
  maxLength?: number;
  tooltip?: boolean;
  className?: string;
  children?: string;
  multiline?: boolean;
}

export const TruncatedText: React.FC<TruncatedTextProps> = ({
  text,
  className,
  maxLength = 18,
  tooltip = true,
  children,
  multiline = false,
}) => {
  text = text || children?.toString() || "";
  const isTruncated = text.length > maxLength;
  const truncatedText = isTruncated ? `${text.slice(0, maxLength)}...` : text;

  if (multiline) {
    return (
      <span
        className={cn(className)}
        style={{ whiteSpace: "pre-line" }}
        title={tooltip && isTruncated ? text : undefined}
      >
        {text}
      </span>
    );
  }

  return (
    <span
      className={cn("truncate", className)}
      title={tooltip && isTruncated ? text : undefined}
    >
      {truncatedText}
    </span>
  );
};
