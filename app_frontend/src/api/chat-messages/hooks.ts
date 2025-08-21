import { useState, useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import i18n from '@/i18n';
import {
  createChat,
  deleteChat,
  deleteMessage,
  exportChatMessages,
  getChatMessages,
  getChats,
  getSingleMessage,
  IChatCreated,
  postMessage,
  renameChat,
  updateChat,
} from './api-requests';
import { messageKeys } from './keys';
import { IChat, IChatMessage, IPostMessageContext, IUserMessage } from './types';
import { useNavigate } from 'react-router-dom';
import { generateChatRoute } from '@/pages/routes';

const POLL_INTERVAL = 1000;

export interface IFetchMessagesParams {
  chatId?: string;
}

export const useFetchAllMessages = ({ chatId }: IFetchMessagesParams) => {
  const queryResult = useQuery<IChatMessage[]>({
    queryKey: messageKeys.messages(chatId),
    queryFn: ({ signal }) => (chatId ? getChatMessages({ signal, chatId }) : Promise.resolve([])),
    enabled: !!chatId,
  });

  return queryResult;
};

export const usePollInProgressMessage = ({ chatId }: { chatId?: string }) => {
  const queryClient = useQueryClient();
  const currentMessages = queryClient.getQueryData<IChatMessage[]>(messageKeys.messages(chatId));
  const inProgressMessageId = (currentMessages || []).find(msg => msg.in_progress)?.id;

  // Poll for single in-progress message
  const { data: polledMessage } = useQuery<IChatMessage | null>({
    queryKey: messageKeys.singleMessage(chatId, inProgressMessageId),
    queryFn: ({ signal }) =>
      getSingleMessage({ signal, chatId: chatId!, messageId: inProgressMessageId! }),
    enabled: !!inProgressMessageId && !!chatId,
    refetchInterval: query => (query.state?.data?.in_progress ? POLL_INTERVAL : false),
  });

  useEffect(() => {
    if (polledMessage && chatId) {
      // Update the specific message in cache
      queryClient.setQueryData<IChatMessage[]>(messageKeys.messages(chatId), cachedMessages =>
        (cachedMessages || []).map(msg => (msg.id === polledMessage.id ? polledMessage : msg))
      );

      // If just completed, trigger refresh for list
      if (!polledMessage.in_progress) {
        queryClient.invalidateQueries({ queryKey: messageKeys.messages(chatId) });
      }
    }
  }, [polledMessage, queryClient, chatId]);
};

export const usePostMessage = () => {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const mutation = useMutation<IChatCreated, Error, IUserMessage, IPostMessageContext>({
    mutationFn: ({ message, chatId, enableChartGeneration, enableBusinessInsights, dataSource }) =>
      postMessage({
        message,
        chatId,
        enableChartGeneration,
        enableBusinessInsights,
        dataSource,
      }),
    onMutate: async ({ message, chatId }) => {
      const messagesKey = messageKeys.messages(chatId);

      // Save previous chats data for rollback if needed
      const previousMessages = queryClient.getQueryData<IChatMessage[]>(messagesKey) || [];
      const previousChats = !chatId
        ? queryClient.getQueryData<IChat[]>(messageKeys.chats) || []
        : undefined;

      // Optimistically update the UI by adding the new message
      const optimisticUpdateMessage: IChatMessage = {
        role: 'user',
        content: message,
        components: [],
        in_progress: true,
        created_at: new Date().toISOString(),
      };
      queryClient.setQueryData(messagesKey, [...previousMessages, optimisticUpdateMessage]);

      return { previousMessages, messagesKey, previousChats };
    },
    onError: (_error, variables, context) => {
      // Restore previous messages
      if (context?.previousMessages && context?.messagesKey) {
        queryClient.setQueryData(context.messagesKey, context.previousMessages);
      }

      // Restore previous chats if this was a new chat operation
      if (!variables.chatId && context?.previousChats) {
        queryClient.setQueryData(messageKeys.chats, context.previousChats);
      }
    },
    onSuccess: (data, variables) => {
      const messages = data?.messages;
      const chatId = data?.id;

      // Update messages to include the optimistically created one with the fresh list from server
      queryClient.setQueryData(messageKeys.messages(chatId), messages);

      // When redirecting from InitialPrompt(no chatId)
      if (!variables.chatId) {
        // Optimistically add newly created chat to the chats list
        queryClient.setQueryData<IChat[]>(messageKeys.chats, (oldData = []) => {
          const chatExists = oldData.some(chat => chat.id === data.id);
          if (chatExists) {
            return oldData;
          }

          return [
            {
              id: data.id,
              name: data.name,
              created_at: new Date().toISOString(),
            } as IChat,
            ...oldData,
          ];
        });
        navigate(generateChatRoute(chatId));
      }
    },
  });

  return mutation;
};

export interface IDeleteMessagesResult {
  success: boolean;
}

export interface IDeleteMessageParams {
  messageId: string;
  chatId?: string;
}

export const useDeleteMessage = () => {
  const queryClient = useQueryClient();
  const mutation = useMutation<
    IChatMessage[],
    Error,
    IDeleteMessageParams,
    { previousMessages: IChatMessage[]; messagesKey: string[] }
  >({
    mutationFn: ({ messageId }) => {
      return deleteMessage({ messageId });
    },
    onMutate: async ({ messageId, chatId }) => {
      if (!chatId) {
        return { previousMessages: [], messagesKey: [] };
      }

      const messagesKey = messageKeys.messages(chatId);
      await queryClient.cancelQueries({ queryKey: messagesKey });

      const previousMessages = queryClient.getQueryData<IChatMessage[]>(messagesKey) || [];

      // Optimistically update the UI by removing the message
      queryClient.setQueryData<IChatMessage[]>(messagesKey, oldData =>
        (oldData || []).filter(m => m.id !== messageId)
      );

      return { previousMessages, messagesKey };
    },
    onError: (error, _, context) => {
      console.error('Error deleting message:', error);

      if (context?.previousMessages && context?.messagesKey) {
        queryClient.setQueryData(context.messagesKey, context.previousMessages);
      }
    },
    onSuccess: (_, variables) => {
      if (variables.chatId) {
        queryClient.invalidateQueries({
          queryKey: messageKeys.messages(variables.chatId),
        });
      }
    },
  });

  return mutation;
};

export const useFetchAllChats = <TData = IChat[]>(options = {}) => {
  const queryResult = useQuery<IChat[], unknown, TData>({
    queryKey: messageKeys.chats,
    queryFn: ({ signal }) => getChats({ signal, limit: 100 }),
    ...options,
  });

  return queryResult;
};

export interface ICreateChatParams {
  name: string;
  dataSource: string;
}

export const useCreateChat = () => {
  const queryClient = useQueryClient();
  const mutation = useMutation<IChat, Error, ICreateChatParams>({
    mutationFn: ({ name, dataSource }) => createChat({ name, dataSource }),
    onMutate: async () => {
      await queryClient.cancelQueries({ queryKey: messageKeys.chats });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: messageKeys.chats });
    },
  });

  return mutation;
};

export interface IDeleteChatParams {
  chatId: string;
}

export const useDeleteChat = ({ onSuccess }: { onSuccess?: () => void }) => {
  const queryClient = useQueryClient();
  const mutation = useMutation<void, Error, IDeleteChatParams, { previousChats: IChat[] }>({
    mutationFn: ({ chatId }) => deleteChat({ chatId }),
    onMutate: async ({ chatId }) => {
      await queryClient.cancelQueries({ queryKey: messageKeys.chats });

      const previousChats = queryClient.getQueryData<IChat[]>(messageKeys.chats) || [];

      queryClient.setQueryData<IChat[]>(messageKeys.chats, oldData => {
        if (!oldData) return [];
        return oldData.filter(chat => chat.id !== chatId);
      });

      return { previousChats };
    },
    onError: (_, __, context) => {
      if (context?.previousChats) {
        queryClient.setQueryData(messageKeys.chats, context.previousChats);
      }
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: messageKeys.chats });
      // Invalidate the specific chat messages
      queryClient.invalidateQueries({
        queryKey: messageKeys.messages(variables.chatId),
      });
      onSuccess?.();
    },
  });

  return mutation;
};

export interface IRenameChatParams {
  chatId: string;
  name: string;
}

export const useRenameChat = () => {
  const queryClient = useQueryClient();
  const mutation = useMutation<void, Error, IRenameChatParams, { previousChats: IChat[] }>({
    mutationFn: ({ chatId, name }) => renameChat({ chatId, name }),
    onMutate: async ({ chatId, name }) => {
      await queryClient.cancelQueries({ queryKey: messageKeys.chats });

      const previousChats = queryClient.getQueryData<IChat[]>(messageKeys.chats) || [];

      // Optimistically update the chat name
      queryClient.setQueryData<IChat[]>(messageKeys.chats, oldData => {
        if (!oldData) return [];
        return oldData.map(chat => (chat.id === chatId ? { ...chat, name } : chat));
      });

      return { previousChats };
    },
    onError: (_, __, context) => {
      if (context?.previousChats) {
        queryClient.setQueryData(messageKeys.chats, context.previousChats);
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: messageKeys.chats });
    },
  });

  return mutation;
};

export interface IUpdateChatDataSourceParams {
  chatId: string;
  dataSource: string;
}

export const useUpdateChatDataSource = () => {
  const queryClient = useQueryClient();
  const mutation = useMutation<
    void,
    Error,
    IUpdateChatDataSourceParams,
    { previousChats: IChat[] }
  >({
    mutationFn: ({ chatId, dataSource }) => updateChat({ chatId, dataSource }),
    onMutate: async ({ chatId, dataSource }) => {
      await queryClient.cancelQueries({ queryKey: messageKeys.chats });

      const previousChats = queryClient.getQueryData<IChat[]>(messageKeys.chats) || [];

      // Optimistically update the chat data source
      queryClient.setQueryData<IChat[]>(messageKeys.chats, oldData => {
        if (!oldData) return [];
        return oldData.map(chat =>
          chat.id === chatId
            ? {
                ...chat,
                data_source: dataSource,
              }
            : chat
        );
      });

      return { previousChats };
    },
    onError: (_, __, context) => {
      if (context?.previousChats) {
        queryClient.setQueryData(messageKeys.chats, context.previousChats);
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: messageKeys.chats });
    },
  });

  return mutation;
};

export interface IExportChatParams {
  chatId: string;
  messageId?: string | null;
}

export const useExport = () => {
  const [isLoading, setIsLoading] = useState(false);

  const exportChat = async ({ chatId, messageId }: IExportChatParams) => {
    setIsLoading(true);

    try {
      const response = await exportChatMessages({ chatId, messageId });

      const filename = messageId
        ? i18n.t('chat_{{chatId}}_message_{{messageId}}.xlsx', { chatId, messageId })
        : i18n.t('chat_{{chatId}}_messages.xlsx', { chatId });

      const url = window.URL.createObjectURL(response.data);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download error:', error);
      toast.error(i18n.t('There was a problem downloading the file.'));
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  return {
    exportChat,
    isLoading,
  };
};
