import React from 'react';

interface MessageContainerProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  testId?: string;
}

export const MessageContainer: React.FC<MessageContainerProps> = React.memo(
  ({ children, testId, ...props }) => {
    return (
      <div
        className="p-3 bg-card rounded flex-col justify-start items-start gap-3 flex mb-8 mr-2"
        data-testid={testId}
        {...props}
      >
        {children}
      </div>
    );
  }
);
