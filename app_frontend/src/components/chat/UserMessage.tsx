import React, { useRef, useEffect } from 'react';
import { MessageHeader } from './MessageHeader';
import { IChatMessage } from '@/api/chat-messages/types';
import { Loading } from './Loading';

interface UserMessageProps {
  message: IChatMessage;
  chatId: string;
  testId?: string;
}

export const UserMessage: React.FC<UserMessageProps> = ({ message, chatId, testId }) => {
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
    >
      <MessageHeader messageId={message.id} chatId={chatId} />
      <div className="self-stretch text-sm font-normal leading-tight whitespace-pre-line">
        {message.content}
      </div>
      {message.in_progress && (
        <div className="flex w-full justify-start">
          <Loading />
        </div>
      )}
    </div>
  );
};
