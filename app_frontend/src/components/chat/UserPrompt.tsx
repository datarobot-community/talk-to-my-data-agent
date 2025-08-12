import { useMemo } from 'react';
import { PromptInput } from '@/components/ui-custom/prompt-input';
import { useFetchAllChats } from '@/api/chat-messages/hooks';
import { useAppState } from '@/state';
import { useTranslation } from '@/i18n';
import { DATA_SOURCES } from '@/constants/dataSources';
import { useChatMessages } from '@/hooks/useChatMessages';

export const UserPrompt = ({
  chatId,
  allowedDataSources,
  testId,
}: {
  chatId?: string;
  allowedDataSources?: string[];
  testId?: string;
}) => {
  const { t } = useTranslation();
  const {
    enableChartGeneration,
    enableBusinessInsights,
    dataSource: globalDataSource,
  } = useAppState();
  const { data: chats } = useFetchAllChats();
  const { hasInProgressMessages, sendMessage } = useChatMessages(chatId);
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

  return (
    <PromptInput
      sendButtonArrangement="append"
      onSend={(message: string) =>
        sendMessage(message, {
          enableChartGeneration,
          enableBusinessInsights,
          dataSource: chatDataSource,
        })
      }
      isProcessing={hasInProgressMessages}
      placeholder={
        isDataUploadRequired
          ? t('Please upload and process data using the sidebar before starting the chat')
          : t('Ask another question about your datasets.')
      }
      isDisabled={isDataUploadRequired}
      testId={testId}
    />
  );
};
