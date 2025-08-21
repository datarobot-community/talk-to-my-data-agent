import React, { useState } from 'react';
import { useDeleteMessage, useExport } from '@/api/chat-messages/hooks';
import { getMessage, getResponseMessage } from '@/api/chat-messages/selectors';
import { UserAvatar, DataRobotAvatar } from './Avatars';
import { Button } from '@/components/ui/button';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faTrash } from '@fortawesome/free-solid-svg-icons/faTrash';
import { faFileArrowDown } from '@fortawesome/free-solid-svg-icons/faFileArrowDown';
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
        : t('Export chat');

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
      <div className="self-stretch justify-between items-center gap-1 inline-flex">
        <div className="grow shrink basis-0 h-6 justify-start items-center gap-2 flex">
          {avatar()}
          <div className="text-sm font-semibold leading-tight">{name}</div>
          <div className="text-xs font-normal leading-[17px]">{date}</div>
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
              <FontAwesomeIcon icon={faFileArrowDown} />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setOpen(true)}
              title={deleteButtonTooltip}
              disabled={isDeleteDisabled}
            >
              <FontAwesomeIcon icon={faTrash} />
            </Button>
          </div>
        )}
      </div>
    </>
  );
};
