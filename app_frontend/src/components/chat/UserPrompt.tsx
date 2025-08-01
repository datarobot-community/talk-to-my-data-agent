import { useMemo } from 'react';
import { PromptInput } from '@/components/ui-custom/prompt-input';
import { usePostMessage, useFetchAllChats } from '@/api/chat-messages/hooks';
import { useAppState } from '@/state';
import { useTranslation } from '@/i18n';
import { DATA_SOURCES } from '@/constants/dataSources';

export const UserPrompt = ({
  chatId,
  isProcessing,
  allowedDataSources,
}: {
  chatId?: string;
  isProcessing?: boolean;
  allowedDataSources?: string[];
}) => {
  const { t } = useTranslation();
  const { mutate } = usePostMessage();
  const {
    enableChartGeneration,
    enableBusinessInsights,
    dataSource: globalDataSource,
  } = useAppState();
  const { data: chats } = useFetchAllChats();
  const isDataUploadRequired = !allowedDataSources?.[0];

  // Find the active chat to get its data source setting
  const activeChat = chatId ? chats?.find(chat => chat.id === chatId) : undefined;
  const chatDataSource = useMemo(() => {
    const dataSource = activeChat?.data_source || globalDataSource;
    // User can only select from the allowed data sources
    return allowedDataSources?.includes(dataSource)
      ? dataSource
      : allowedDataSources?.[0] || DATA_SOURCES.FILE;
  }, [activeChat?.data_source, globalDataSource, allowedDataSources]);

  const sendMessage = (message: string) => {
    mutate({
      message,
      chatId,
      enableChartGeneration,
      enableBusinessInsights,
      dataSource: chatDataSource,
    });
  };

  return (
    <PromptInput
      sendButtonArrangement="append"
      onSend={sendMessage}
      isProcessing={isProcessing}
      placeholder={
        isDataUploadRequired
          ? t('Please upload and process data using the sidebar before starting the chat')
          : t('Ask another question about your datasets.')
      }
      isDisabled={isDataUploadRequired}
    />
  );
};
