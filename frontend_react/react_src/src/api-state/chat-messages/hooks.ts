import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createChat,
  deleteAllMessages,
  deleteChat,
  deleteMessage,
  getChatMessages,
  getChats,
  IChatCreated,
  postMessage,
  renameChat,
  updateChat,
} from "./api-requests";
import { messageKeys } from "./keys";
import {
  IChat,
  IChatMessage,
  IPostMessageContext,
  IUserMessage,
} from "./types";
import { useNavigate } from "react-router-dom";
import { generateChatRoute } from "@/pages/routes";

export interface IFetchMessagesParams {
  limit?: number;
  chatId?: string;
}

export const useFetchAllMessages = ({
  chatId,
  limit = 100,
}: IFetchMessagesParams) => {
  const queryResult = useQuery<IChatMessage[]>({
    queryKey: messageKeys.messages(chatId),
    queryFn: ({ signal }) =>
      chatId ? getChatMessages({ signal, limit, chatId }) : Promise.resolve([]),
    refetchInterval: (query) =>
      !query ||
      // query.state?.data?.length === 0 ||
      query.state?.data?.some((d) => d.in_progress)
        ? 5000
        : false,
  });

  return queryResult;
};

export const usePostMessage = () => {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const mutation = useMutation<
    IChatCreated,
    Error,
    IUserMessage,
    IPostMessageContext
  >({
    mutationFn: ({
      message,
      chatId,
      enableChartGeneration,
      enableBusinessInsights,
      dataSource,
    }) =>
      postMessage({
        message,
        chatId,
        enableChartGeneration,
        enableBusinessInsights,
        dataSource,
      }),
    onMutate: async ({ message, chatId }) => {
      const messagesKey = messageKeys.messages(chatId);
      await queryClient.cancelQueries({ queryKey: messagesKey });

      // If this is a new chat, also cancel chats query
      if (!chatId) {
        await queryClient.cancelQueries({ queryKey: messageKeys.chats });
      }

      const previousMessages =
        queryClient.getQueryData<IChatMessage[]>(messagesKey) || [];

      // Save previous chats data for rollback if needed
      const previousChats = !chatId
        ? queryClient.getQueryData<IChat[]>(messageKeys.chats) || []
        : undefined;

      const newUserMessage: IChatMessage = {
        role: "user",
        content: message,
        components: [],
        in_progress: true,
        created_at: new Date().toISOString(),
      };

      queryClient.setQueryData(messagesKey, [
        ...previousMessages,
        newUserMessage,
      ]);

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
      // When redirecting from InitialPrompt (no chatId), don't invalidate queries
      // as we'll navigate to the new chat which will fetch the messages
      if (!variables.chatId) {
        // Set the chat messages data directly in the cache to avoid loading state
        queryClient.setQueryData<IChatMessage[]>(
          messageKeys.messages(data.id),
          (oldData = []) => [
            ...oldData,
            {
              role: "user",
              content: variables.message,
              components: [],
              in_progress: true,
              created_at: new Date().toISOString(),
            },
          ]
        );

        // Update the chats list cache with the new chat
        queryClient.setQueryData<IChat[]>(messageKeys.chats, (oldData = []) => [
          {
            id: data.id,
            name: "New Chat",
            created_at: new Date().toISOString(),
          } as IChat,
          ...oldData,
        ]);

        // Navigate to the new chat
        navigate(generateChatRoute(data.id));
      } else {
        // For existing chats, invalidate as usual
        queryClient.invalidateQueries({
          queryKey: messageKeys.messages(variables.chatId),
        });
        queryClient.invalidateQueries({ queryKey: messageKeys.chats });
      }
    },
  });

  return mutation;
};

export interface IDeleteMessagesResult {
  success: boolean;
}

export const useDeleteAllMessages = () => {
  const queryClient = useQueryClient();
  const mutation = useMutation<
    void,
    Error,
    { chatId: string },
    { previousMessages: IChatMessage[]; messagesKey: string[] }
  >({
    mutationFn: ({ chatId }) => deleteAllMessages({ chatId }),
    onMutate: async ({ chatId }) => {
      const messagesKey = messageKeys.messages(chatId);
      await queryClient.cancelQueries({ queryKey: messagesKey });
      const previousMessages =
        queryClient.getQueryData<IChatMessage[]>(messagesKey) || [];
      queryClient.setQueryData(messagesKey, []);
      return { previousMessages, messagesKey };
    },
    onError: (_, __, context) => {
      if (context?.previousMessages && context?.messagesKey) {
        queryClient.setQueryData(context.messagesKey, context.previousMessages);
      }
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: messageKeys.messages(variables?.chatId),
      });
    },
  });
  return mutation;
};

export interface IDeleteMessageParams {
  index: number;
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
    mutationFn: ({ index, chatId }) => deleteMessage({ index, chatId }),
    onMutate: async ({ index, chatId }) => {
      const messagesKey = messageKeys.messages(chatId);
      await queryClient.cancelQueries({ queryKey: messagesKey });

      const previousMessages =
        queryClient.getQueryData<IChatMessage[]>(messagesKey) || [];

      // Optimistically update the UI by removing the message at the index
      // and its preceding/following pair
      queryClient.setQueryData<IChatMessage[]>(messagesKey, (oldData) => {
        if (!oldData) return [];

        // Create a copy to avoid mutating the original
        const newData = [...oldData];

        // Find the target message
        const targetMessage = newData[index];
        if (!targetMessage) return newData;

        // For assistant messages, also remove the preceding user message
        if (targetMessage.role === "assistant" && index > 0) {
          // Find the preceding user message
          let precedingIndex = index - 1;
          while (precedingIndex >= 0) {
            if (newData[precedingIndex].role === "user") {
              // Remove both messages (higher index first to avoid shifting issues)
              newData.splice(Math.max(index, precedingIndex), 1);
              newData.splice(Math.min(index, precedingIndex), 1);
              break;
            }
            precedingIndex -= 1;
          }
        }
        // For user messages, also remove the following assistant message
        else if (targetMessage.role === "user") {
          // Find the following assistant message
          let followingIndex = index + 1;
          while (followingIndex < newData.length) {
            if (newData[followingIndex].role === "assistant") {
              // Remove both messages (higher index first to avoid shifting issues)
              newData.splice(Math.max(index, followingIndex), 1);
              newData.splice(Math.min(index, followingIndex), 1);
              break;
            }
            followingIndex += 1;
          }

          // If no following assistant message was found, just remove the user message
          if (followingIndex >= newData.length) {
            newData.splice(index, 1);
          }
        }

        return newData;
      });

      return { previousMessages, messagesKey };
    },
    onError: (_, __, context) => {
      if (context?.previousMessages && context?.messagesKey) {
        queryClient.setQueryData(context.messagesKey, context.previousMessages);
      }
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: messageKeys.messages(variables.chatId),
      });
    },
  });

  return mutation;
};

export const useFetchAllChats = ({ limit = 100 } = {}) => {
  const queryResult = useQuery({
    queryKey: messageKeys.chats,
    queryFn: ({ signal }) => getChats({ signal, limit }),
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

export const useDeleteChat = () => {
  const queryClient = useQueryClient();
  const mutation = useMutation<
    void,
    Error,
    IDeleteChatParams,
    { previousChats: IChat[] }
  >({
    mutationFn: ({ chatId }) => deleteChat({ chatId }),
    onMutate: async ({ chatId }) => {
      await queryClient.cancelQueries({ queryKey: messageKeys.chats });

      const previousChats =
        queryClient.getQueryData<IChat[]>(messageKeys.chats) || [];

      queryClient.setQueryData<IChat[]>(messageKeys.chats, (oldData) => {
        if (!oldData) return [];
        return oldData.filter((chat) => chat.id !== chatId);
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
  const mutation = useMutation<
    void,
    Error,
    IRenameChatParams,
    { previousChats: IChat[] }
  >({
    mutationFn: ({ chatId, name }) => renameChat({ chatId, name }),
    onMutate: async ({ chatId, name }) => {
      await queryClient.cancelQueries({ queryKey: messageKeys.chats });

      const previousChats =
        queryClient.getQueryData<IChat[]>(messageKeys.chats) || [];

      // Optimistically update the chat name
      queryClient.setQueryData<IChat[]>(messageKeys.chats, (oldData) => {
        if (!oldData) return [];
        return oldData.map((chat) =>
          chat.id === chatId ? { ...chat, name } : chat
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

      const previousChats =
        queryClient.getQueryData<IChat[]>(messageKeys.chats) || [];

      // Optimistically update the chat data source
      queryClient.setQueryData<IChat[]>(messageKeys.chats, (oldData) => {
        if (!oldData) return [];
        return oldData.map((chat) =>
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
