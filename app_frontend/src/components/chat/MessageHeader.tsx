import React, { useState } from 'react';
import { useDeleteMessage, useExport, useUpdateMessageFeedback } from '@/api/chat-messages/hooks';
import { getMessage, getResponseMessage } from '@/api/chat-messages/selectors';
import { UserAvatar, DataRobotAvatar } from './Avatars';
import { Button } from '@/components/ui/button';
import { Trash2, FileDown, ThumbsUp, ThumbsDown } from 'lucide-react';
import { useTranslation } from '@/i18n';
import { ConfirmDialog } from '../ui-custom/confirm-dialog';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
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
  const [isFeedbackDialogOpen, setIsFeedbackDialogOpen] = useState(false);
  const [feedbackText, setFeedbackText] = useState('');
  const [shouldSubmitNegativeOnClose, setShouldSubmitNegativeOnClose] = useState(false);
  const { mutate: deleteMessage, isPending: isDeleting } = useDeleteMessage();
  const { exportChat, isLoading: isExporting } = useExport();
  const { mutate: updateMessageFeedback, isPending: isUpdatingFeedback } =
    useUpdateMessageFeedback();

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

  const userMessageFailed = isUserMessage && !!message.error;
  const isDeleteDisabled = !responseMessage && !userMessageFailed;
  const deleteButtonTooltip = isDeleteDisabled
    ? t('Cannot delete message without response')
    : userMessageFailed
      ? t('Delete message')
      : t('Delete message and response');

  const handleThumbsUp = () => {
    if (!message.id) {
      return;
    }

    updateMessageFeedback({
      messageId: message.id,
      chatId,
      userRating: 1,
      userFeedback: '', // empty feedback to clear potential prior negative feedback.
    });
  };

  const handleThumbsDown = () => {
    setFeedbackText('');
    setShouldSubmitNegativeOnClose(true);
    setIsFeedbackDialogOpen(true);
  };

  const handleFeedbackSubmit = () => {
    if (!message.id) {
      return;
    }

    updateMessageFeedback({
      messageId: message.id,
      chatId,
      userRating: -1,
      userFeedback: feedbackText,
    });
    setShouldSubmitNegativeOnClose(false);
    setIsFeedbackDialogOpen(false);
  };

  const handleFeedbackKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter') {
      return;
    }

    const isSubmitShortcutPressed = event.metaKey || event.ctrlKey;

    if (!isSubmitShortcutPressed) {
      return;
    }

    event.preventDefault();
    handleFeedbackSubmit();
  };

  const handleFeedbackDialogOpenChange = (open: boolean) => {
    if (!open && shouldSubmitNegativeOnClose && message.id) {
      updateMessageFeedback({
        messageId: message.id,
        chatId,
        userRating: -1,
      });
      setShouldSubmitNegativeOnClose(false);
    }

    setIsFeedbackDialogOpen(open);
  };

  const isPositiveSelected = message.user_rating === 1;
  const isNegativeSelected = message.user_rating === -1;
  const feedbackDisabled = !message.id || isUpdatingFeedback || message.role !== 'assistant';

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
      <Dialog open={isFeedbackDialogOpen} onOpenChange={handleFeedbackDialogOpenChange}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{t('What could have been better?')}</DialogTitle>
          </DialogHeader>
          <textarea
            value={feedbackText}
            onChange={event => setFeedbackText(event.target.value)}
            onKeyDown={handleFeedbackKeyDown}
            className="min-h-24 w-full rounded border border-border bg-input px-3 py-2 text-sm outline-none hover:border-muted-foreground focus:border-accent"
            placeholder={t('Describe the issue (optional)')}
            data-testid="message-feedback-textarea"
          />
          <DialogFooter>
            <Button variant="secondary" onClick={handleFeedbackSubmit}>
              {t('Submit')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <div className="inline-flex items-center justify-between gap-1 self-stretch">
        <div className="flex h-6 shrink grow basis-0 items-center justify-start gap-2">
          {avatar()}
          <div className="mn-label-large">{name}</div>
          <div className="body-secondary">{date}</div>
        </div>
        {isUserMessage && (
          <div className="flex items-center">
            {!userMessageFailed && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => exportMessage(messageId)}
                title={exportButtonTooltip}
                disabled={isExportDisabled}
              >
                <FileDown />
              </Button>
            )}
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
        {!isUserMessage && (
          <div className="flex items-center gap-0">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleThumbsUp}
              disabled={feedbackDisabled}
              title={t('Leave positive feedback')}
              testId="message-feedback-thumbs-up"
              aria-pressed={isPositiveSelected}
              className={
                isPositiveSelected
                  ? 'text-success drop-shadow-[0_0_6px_hsl(var(--success)/0.85)] hover:drop-shadow-none'
                  : undefined
              }
            >
              <ThumbsUp />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleThumbsDown}
              disabled={feedbackDisabled}
              title={t('Leave negative feedback')}
              testId="message-feedback-thumbs-down"
              aria-pressed={isNegativeSelected}
              className={
                isNegativeSelected
                  ? 'text-destructive drop-shadow-[0_0_6px_hsl(var(--destructive)/0.85)] hover:drop-shadow-none'
                  : undefined
              }
            >
              <ThumbsDown />
            </Button>
          </div>
        )}
      </div>
    </>
  );
};
