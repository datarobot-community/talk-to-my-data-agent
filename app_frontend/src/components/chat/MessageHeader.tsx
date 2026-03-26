import React, { useState } from 'react';
import { useDeleteMessage, useExport } from '@/api/chat-messages/hooks';
import { getMessage, getResponseMessage } from '@/api/chat-messages/selectors';
import { UserAvatar, DataRobotAvatar } from './Avatars';
import { Button } from '@/components/ui/button';
import { Trash2, FileDown } from 'lucide-react';
import { useTranslation } from '@/i18n';
import { ConfirmDialog } from '../ui-custom/confirm-dialog';
import { formatMessageDate } from './utils';
import { toast } from 'sonner';
import { IChatMessage } from '@/api/chat-messages/types';

interface MessageHeaderProps {
  messageId?: string;
  chatId: string;
  messages: IChatMessage[];
}

export const MessageHeader: React.FC<MessageHeaderProps> = ({ messageId, chatId, messages }) => {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const { mutate: deleteMessage, isPending: isDeleting } = useDeleteMessage();
  const { exportChat, isLoading: isExporting } = useExport();

  // If no message is found by id - get optimistically created one (it has no id yet).
  const message = getMessage(messages, messageId) || messages?.[0];
  if (!message) {
    return null;
  }

  const deleteMessagePair = (userMessageId: string | undefined | null) => {
    const userMessage = getMessage(messages, userMessageId);
    if (!userMessage || userMessage.role !== 'user' || !userMessage.id) {
      throw new Error('User message not found');
    }

    // Delete response first (if exists), then user message
    const responseMessage = getResponseMessage(messages, userMessageId);
    if (responseMessage?.id && responseMessage !== userMessage) {
      deleteMessage({ messageId: responseMessage.id, chatId });
    }
    deleteMessage({ messageId: userMessage.id, chatId });
  };

  const exportMessage = (messageId: string | undefined | null) => {
    if (!chatId) {
      toast.error(t('Chat ID not found'));
      return;
    }
    if (!messageId) {
      toast.error(t('Message ID not found for export'));
      return;
    }

    exportChat({ chatId, messageId });
  };

  const isUserMessage = message.role === 'user';
  const avatar = isUserMessage ? UserAvatar : DataRobotAvatar;
  const name = isUserMessage ? t('You') : t('DataRobot');

  const date = formatMessageDate(message.created_at);

  const responseMessage = getResponseMessage(messages, messageId);
  const responseInProgress = !!responseMessage?.in_progress;
  const responseFailing = !!responseMessage?.error;

  const isExportDisabled = isExporting || responseInProgress || responseFailing || !responseMessage;
  const exportButtonTooltip = responseFailing
    ? t('Cannot export chat with errors')
    : isExporting
      ? t('Exporting...')
      : responseInProgress || !responseMessage
        ? t('Wait for agent to finish responding')
        : t('Export prompt and response');

  const isDeleteDisabled = !responseMessage;
  const deleteButtonTooltip = isDeleteDisabled
    ? t('Cannot delete message without response')
    : t('Delete message and response');

  return (
    <>
      <ConfirmDialog
        open={open}
        onOpenChange={setOpen}
        title={t('Delete message')}
        description={t('Are you sure you want to delete this message?')}
        onConfirm={() => deleteMessagePair(messageId)}
        confirmText={t('Delete')}
        cancelText={t('Cancel')}
        variant="destructive"
        isLoading={isDeleting}
      />
      <div className="inline-flex items-center justify-between gap-1 self-stretch">
        <div className="flex h-6 shrink grow basis-0 items-center justify-start gap-2">
          {avatar()}
          <div className="mn-label-large">{name}</div>
          <div className="body-secondary">{date}</div>
        </div>
        {isUserMessage && (
          <div className="flex items-center">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => exportMessage(messageId)}
              title={exportButtonTooltip}
              disabled={isExportDisabled}
            >
              <FileDown />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setOpen(true)}
              title={deleteButtonTooltip}
              disabled={isDeleteDisabled}
            >
              <Trash2 />
            </Button>
          </div>
        )}
      </div>
    </>
  );
};
