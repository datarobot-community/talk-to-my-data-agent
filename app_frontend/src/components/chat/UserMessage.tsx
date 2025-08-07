import React, { useEffect, useRef } from 'react';
import { MessageHeader } from './MessageHeader';
import { formatMessageDate } from './utils';
import { useDeleteMessage, useExport } from '@/api/chat-messages/hooks';
import { useTranslation } from '@/i18n';
import { IChatMessage } from '@/api/chat-messages/types';

interface UserMessageProps {
  messageId?: string;
  timestamp?: string;
  message: string;
  chatId?: string;
  testId?: string;
  responseMessage?: IChatMessage;
}

export const UserMessage: React.FC<UserMessageProps> = ({
  chatId,
  messageId,
  timestamp,
  message,
  testId,
  responseMessage,
}) => {
  const ref = useRef<HTMLDivElement>(null);
  const { mutate: deleteMessage, isPending: isDeleting } = useDeleteMessage();
  const { exportChat, isLoading: isExporting } = useExport();
  const { t } = useTranslation();
  useEffect(() => {
    // When being somewhere in the middle of the chat and asking question scroll to it
    ref.current?.scrollIntoView({ behavior: 'smooth' });
  }, [message]);

  const displayDate = timestamp ? formatMessageDate(timestamp) : '';

  const handleDelete = () => {
    if (messageId) {
      deleteMessage({
        chatId: chatId,
        messageId: messageId,
      });
    }
    if (responseMessage?.id) {
      deleteMessage({
        chatId: chatId,
        messageId: responseMessage.id,
      });
    }
  };

  const handleExport = () => {
    if (chatId) {
      exportChat({
        chatId,
        messageId,
      });
    }
  };

  return (
    <div
      className="p-3 bg-card rounded flex-col justify-start items-start gap-3 flex mb-2.5 mr-2"
      ref={ref}
      data-testid={testId}
    >
      <MessageHeader
        name={t('You')}
        date={displayDate}
        onDelete={handleDelete}
        onExport={handleExport}
        isDeleting={isDeleting}
        isExporting={isExporting}
        isResponseInProgress={responseMessage?.in_progress}
        isResponseFailing={!!responseMessage?.error}
      />
      <div className="self-stretch text-sm font-normal leading-tight whitespace-pre-line">
        {message}
      </div>
    </div>
  );
};
