import React, { useRef, useEffect } from 'react';
import { MessageHeader } from './MessageHeader';
import { IChatMessage } from '@/api/chat-messages/types';
import { Loading } from './Loading';

interface UserMessageProps {
  message: IChatMessage;
  chatId: string;
  messages: IChatMessage[];
  testId?: string;
}

export const UserMessage: React.FC<UserMessageProps> = ({ message, chatId, messages, testId }) => {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // When being somewhere in the middle of the chat and asking question scroll to it
    ref.current?.scrollIntoView({ behavior: 'smooth' });
  }, [message.content]);

  return (
    <div
      className="mr-2 mb-2.5 flex flex-col items-start justify-start gap-3 rounded bg-card p-3"
      ref={ref}
      data-testid={testId}
      key={message.id}
    >
      <MessageHeader messageId={message.id} chatId={chatId} messages={messages} />
      <div className="self-stretch body whitespace-pre-line">{message.content}</div>
      {message.in_progress && (
        <div className="flex w-full justify-start">
          <Loading />
        </div>
      )}
      {message.error && (
        <div className="max-h-[300px] max-w-full overflow-x-auto overflow-y-auto">
          <span className="text-sm text-destructive">{message.error}</span>
        </div>
      )}
    </div>
  );
};
