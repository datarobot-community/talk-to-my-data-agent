import { useMemo } from 'react';
import { PromptInput } from '@/components/ui-custom/prompt-input';
import chatMidnight from '@/assets/chat-midnight.svg';
import chatLight from '@/assets/chat-light.svg';
import { usePostMessage } from '@/api/chat-messages/hooks';
import { useTranslation } from '@/i18n';
import { useAppState } from '@/state/hooks';
import { DATA_SOURCES } from '@/constants/dataSources';
import type { IChat } from '@/api/chat-messages/types';
import { useTheme } from '@/theme/theme-provider';

export const InitialPrompt = ({
  chatId,
  allowedDataSources,
  testId,
  activeChat,
}: {
  allowedDataSources?: string[];
  chatId?: string;
  testId?: string;
  activeChat?: IChat;
}) => {
  const { t } = useTranslation();
  const { theme } = useTheme();
  const {
    enableChartGeneration,
    enableBusinessInsights,
    dataSource: globalDataSource,
  } = useAppState();
  const { mutate: sendMessage } = usePostMessage();
  const isDisabled = !allowedDataSources?.[0];

  const chatDataSource = useMemo(() => {
    const dataSource = activeChat?.data_source || globalDataSource;
    // User can only select from the allowed data sources
    return allowedDataSources?.includes(dataSource)
      ? dataSource
      : allowedDataSources?.[0] || DATA_SOURCES.FILE;
  }, [activeChat?.data_source, globalDataSource, allowedDataSources]);

  return (
    <div className="flex-1 flex flex-col p-4" data-testid={testId}>
      <div className="flex flex-col flex-1 items-center justify-center">
        <div className="w-[400px] flex flex-col flex-1 items-center justify-center">
          <img src={theme === 'dark' ? chatMidnight : chatLight} alt="" />
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
            chatId={chatId}
            sendButtonArrangement="append"
            onSend={(message: string) =>
              sendMessage({
                message,
                chatId,
                enableChartGeneration,
                enableBusinessInsights,
                dataSource: chatDataSource,
              })
            }
            isDisabled={isDisabled}
            testId="initial-prompt-input"
            placeholder={t('Ask another question about your datasets.')}
            autoFocus
          />
        </div>
      </div>
    </div>
  );
};
