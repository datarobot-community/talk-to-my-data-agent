import { useMemo } from 'react';
import { PromptInput } from '@/components/ui-custom/prompt-input';
import chatMidnight from '@/assets/chat-midnight.svg';
import { usePostMessage, useFetchAllChats } from '@/api/chat-messages/hooks';
import { useTranslation } from '@/i18n';
import { useAppState } from '@/state/hooks';
import { DATA_SOURCES } from '@/constants/dataSources';

export const InitialPrompt = ({
  chatId,
  allowedDataSources,
}: {
  allowedDataSources?: string[];
  chatId?: string;
}) => {
  const { t } = useTranslation();
  const {
    enableChartGeneration,
    enableBusinessInsights,
    dataSource: globalDataSource,
  } = useAppState();
  const { data: chats } = useFetchAllChats();
  const { mutate } = usePostMessage();
  const isDisabled = !allowedDataSources?.[0];

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
    <div className="flex-1 flex flex-col p-4">
      <div className="flex flex-col flex-1 items-center justify-center">
        <div className="w-[400px] flex flex-col flex-1 items-center justify-center">
          <img src={chatMidnight} alt="" />
          <h4 className="mb-2 mt-4">
            <strong className=" text-center font-semibold">
              {t('Type a question about your dataset')}
            </strong>
          </h4>
          <p className="text-center mb-10">
            {t(
              "Ask specific questions about your datasets to get insights, generate visualizations, and discover patterns. Include column names and the kind of analysis you're looking for to get more accurate results."
            )}
          </p>
          <PromptInput
            sendButtonArrangement="append"
            onSend={sendMessage}
            isDisabled={isDisabled}
            testId="initial-prompt-input"
            placeholder={t('Ask another question about your datasets.')}
          />
        </div>
      </div>
    </div>
  );
};
