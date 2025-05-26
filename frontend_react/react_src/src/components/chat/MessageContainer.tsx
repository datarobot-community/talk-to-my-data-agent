import React from "react";

interface MessageContainerProps {
  children: React.ReactNode;
}

export const MessageContainer: React.FC<MessageContainerProps> = React.memo(
  ({ children }) => {
    return (
      <div
        className="p-3 bg-card rounded flex-col justify-start items-start gap-3 flex mb-2.5 mr-2"
      >
        {children}
      </div>
    );
  }
);
