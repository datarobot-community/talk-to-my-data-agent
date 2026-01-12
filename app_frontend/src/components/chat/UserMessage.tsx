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
      className="p-3 bg-card rounded flex-col justify-start items-start gap-3 flex mb-2.5 mr-2"
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
        <div className="max-h-[300px] overflow-x-auto overflow-y-auto max-w-full">
          <span className="text-destructive text-sm">{message.error}</span>
        </div>
      )}
    </div>
  );
};
