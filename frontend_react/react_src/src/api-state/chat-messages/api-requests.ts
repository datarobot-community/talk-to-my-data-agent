import apiClient from "../apiClient";
import { IChat, IChatMessage } from "./types";

export interface IGetMessagesParams {
  limit: number;
  chatId: string;
  signal?: AbortSignal;
}

export const getChatMessages = async ({
  signal,
  chatId,
}: IGetMessagesParams): Promise<IChatMessage[]> => {
  const { data } = await apiClient.get<IChatMessage[]>(
    `/v1/chats/${chatId}/messages`,
    {
      signal,
    }
  );

  return data;
};

export interface IPostMessageParams {
  message: string;
  chatId?: string;
  enableChartGeneration?: boolean;
  enableBusinessInsights?: boolean;
  dataSource?: string;
  signal?: AbortSignal;
}

export interface IChatCreated {
  id: string;
  name: string;
  messages: IChatMessage[];
}

export async function postMessage({
  message,
  chatId,
  enableChartGeneration,
  enableBusinessInsights,
  dataSource,
  signal,
}: IPostMessageParams): Promise<IChatCreated> {
  const payload = {
    message: message,
    enable_chart_generation: enableChartGeneration,
    enable_business_insights: enableBusinessInsights,
    data_source: dataSource,
  };

  // If no chatId is provided, create a new chat with the message
  if (!chatId) {
    const { data } = await apiClient.post<IChatCreated>(
      "/v1/chats/messages",
      payload,
      { signal }
    );
    return data;
  }

  // If chatId exists, post to that chat
  const { data } = await apiClient.post<IChatCreated>(
    `/v1/chats/${chatId}/messages`,
    payload,
    { signal }
  );

  return data;
}

export const deleteAllMessages = async ({
  chatId,
}: {
  chatId: string;
}): Promise<void> => {
  const { data } = await apiClient.delete<void>(`/v1/chats/${chatId}/messages`);
  return data;
};

export interface IDeleteMessageParams {
  index: number;
  chatId?: string;
  signal?: AbortSignal;
}

export const deleteMessage = async ({
  index,
  chatId,
  signal,
}: IDeleteMessageParams): Promise<IChatMessage[]> => {
  if (!chatId) {
    throw new Error("Chat ID is required for deleting messages");
  }
  const url = `/v1/chats/${chatId}/messages/${index}`;
  const { data } = await apiClient.delete<IChatMessage[]>(url, { signal });
  return data;
};

export interface IDeleteChatParams {
  chatId: string;
  signal?: AbortSignal;
}

export const deleteChat = async ({
  chatId,
  signal,
}: IDeleteChatParams): Promise<void> => {
  await apiClient.delete(`/v1/chats/${chatId}`, { signal });
  return;
};

export interface IGetChatsParams {
  limit?: number;
  signal?: AbortSignal;
}

export const getChats = async ({
  signal,
}: IGetChatsParams): Promise<IChat[]> => {
  const { data } = await apiClient.get<IChat[]>(`/v1/chats`, {
    signal,
  });
  return data;
};

export interface ICreateChatParams {
  name: string;
  dataSource: string;
  signal?: AbortSignal;
}

export const createChat = async ({
  name,
  dataSource,
  signal,
}: ICreateChatParams): Promise<IChat> => {
  const { data } = await apiClient.post<IChat>(
    "/v1/chats",
    { name, data_source: dataSource },
    { signal }
  );
  return data;
};

export interface IUpdateChatParams {
  chatId: string;
  name?: string;
  dataSource?: string;
  signal?: AbortSignal;
}

export const updateChat = async ({
  chatId,
  name,
  dataSource,
  signal,
}: IUpdateChatParams): Promise<void> => {
  const payload: Record<string, string> = {};
  if (name !== undefined) payload.name = name;
  if (dataSource !== undefined) payload.data_source = dataSource;

  await apiClient.put(`/v1/chats/${chatId}`, payload, { signal });
  return;
};

// Keeping for backward compatibility
export const renameChat = async ({
  chatId,
  name,
  signal,
}: Pick<IUpdateChatParams, "chatId" | "name" | "signal">): Promise<void> => {
  return updateChat({ chatId, name, signal });
};
