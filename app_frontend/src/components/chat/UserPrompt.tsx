import { useMemo } from 'react';
import { PromptInput } from '@/components/ui-custom/prompt-input';
import { usePostMessage } from '@/api/chat-messages/hooks';
import { useAppState } from '@/state';
import { useTranslation } from '@/i18n';
import { DATA_SOURCES } from '@/constants/dataSources';
import type { IChat } from '@/api/chat-messages/types';

export const UserPrompt = ({
  chatId,
  allowedDataSources,
  hasInProgressMessages,
  testId,
  activeChat,
}: {
  chatId?: string;
  allowedDataSources?: string[];
  hasInProgressMessages: boolean;
  testId?: string;
  activeChat?: IChat;
}) => {
  const { t } = useTranslation();

  const {
    enableChartGeneration,
    enableBusinessInsights,
    dataSource: globalDataSource,
  } = useAppState();
  const { mutate: postMessage } = usePostMessage();
  const isDataUploadRequired = !allowedDataSources?.[0];
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
        postMessage({
          message,
          chatId,
          enableChartGeneration,
          enableBusinessInsights,
          dataSource: chatDataSource,
        })
      }
      isProcessing={hasInProgressMessages}
      chatId={chatId}
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
