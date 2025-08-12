import React, { useState } from 'react';
import { useChatMessages } from '@/hooks/useChatMessages';
import { UserAvatar, DataRobotAvatar } from './Avatars';
import { Button } from '@/components/ui/button';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faTrash } from '@fortawesome/free-solid-svg-icons/faTrash';
import { faFileArrowDown } from '@fortawesome/free-solid-svg-icons/faFileArrowDown';
import { useTranslation } from '@/i18n';
import { ConfirmDialog } from '../ui-custom/confirm-dialog';
import { formatMessageDate } from './utils';

interface MessageHeaderProps {
  messageId?: string;
  chatId: string;
}

export const MessageHeader: React.FC<MessageHeaderProps> = ({ messageId, chatId }) => {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const {
    isDeleting,
    isExporting,
    deleteMessagePair,
    exportMessage,
    getMessage,
    getResponseMessage,
  } = useChatMessages(chatId);

  const message = getMessage(messageId);
  if (!message) {
    return null; // Message not found
  }

  const isUserMessage = message.role === 'user';
  const avatar = isUserMessage ? UserAvatar : DataRobotAvatar;
  const name = isUserMessage ? t('You') : t('DataRobot');
  const date = formatMessageDate(message.created_at);

  const responseMessage = getResponseMessage(messageId);
  const responseInProgress = !!responseMessage?.in_progress;
  const responseFailing = !!responseMessage?.error;

  const isExportDisabled = isExporting || responseInProgress || responseFailing;
  const exportButtonTooltip = responseFailing
    ? t('Cannot export chat with errors')
    : isExporting
      ? t('Exporting...')
      : responseInProgress
        ? t('Wait for agent to finish responding')
        : t('Export chat');

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
              title={t('Delete message and response')}
            >
              <FontAwesomeIcon icon={faTrash} />
            </Button>
          </div>
        )}
      </div>
    </>
  );
};
