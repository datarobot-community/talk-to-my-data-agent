import {
  useFetchAllMessages,
  usePostMessage,
  useDeleteMessage,
  useExport,
} from '@/api/chat-messages/hooks';
import { IChatMessage } from '@/api/chat-messages/types';
import { toast } from 'sonner';
import { useTranslation } from '@/i18n';

interface SendMessageOptions {
  enableChartGeneration?: boolean;
  enableBusinessInsights?: boolean;
  dataSource?: string;
}

/**
 * Comprehensive hook for chat message operations and state
 */
export const useChatMessages = (chatId?: string) => {
  const { t } = useTranslation();

  const { data: messages = [], isLoading, error: fetchError } = useFetchAllMessages({ chatId });

  const hasInProgressMessages = messages.some(message => message.in_progress);
  const hasFailedMessages = messages.some(message => message.error);

  const { mutate: postMessageMutate, isPending: isSending, error: sendError } = usePostMessage();

  const { mutate: deleteMutate, isPending: isDeleting, error: deleteError } = useDeleteMessage();

  const { exportChat, isLoading: isExporting } = useExport();

  const getMessage = (messageId: string | undefined | null): IChatMessage | undefined => {
    if (!messageId) {
      return undefined;
    }
    return messages.find(message => message.id === messageId);
  };

  const getResponseMessage = (messageId: string | undefined | null): IChatMessage | undefined => {
    if (!messageId) {
      return undefined;
    }
    const message = getMessage(messageId);
    if (!message) return undefined;

    // If it's already an assistant message, return it
    if (message.role === 'assistant') return message;

    // If it's a user message, find the next assistant message
    const userIndex = messages.findIndex(msg => msg.id === messageId);
    const responseMessage = messages[userIndex + 1];
    return responseMessage?.role === 'assistant' ? responseMessage : undefined;
  };

  // Actions
  const sendMessage = (message: string, options: SendMessageOptions = {}) => {
    if (!message.trim()) {
      toast.error(t('Message cannot be empty'));
      return;
    }

    postMessageMutate({
      message,
      chatId,
      enableChartGeneration: options.enableChartGeneration,
      enableBusinessInsights: options.enableBusinessInsights,
      dataSource: options.dataSource,
    });
  };

  const deleteMessagePair = (userMessageId: string | undefined | null) => {
    const userMessage = getMessage(userMessageId);
    if (!userMessage || userMessage.role !== 'user' || !userMessage.id) {
      throw new Error('User message not found');
    }

    // Delete response first (if exists), then user message
    const responseMessage = getResponseMessage(userMessageId);
    if (responseMessage?.id && responseMessage !== userMessage) {
      deleteMutate({ messageId: responseMessage.id, chatId });
    }
    deleteMutate({ messageId: userMessage.id, chatId });
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

  return {
    // Data
    messages,

    // Loading states
    isLoading,
    isSending,
    isDeleting,
    isExporting,
    hasInProgressMessages,
    hasFailedMessages,

    // Errors
    fetchError,
    sendError,
    deleteError,

    // Utilities
    getMessage,
    getResponseMessage,

    // Actions
    sendMessage,
    deleteMessagePair,
    exportMessage,
  };
};
