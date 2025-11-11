import React from 'react';
import { Zap } from 'lucide-react';
import { IChatMessage } from '@/api/chat-messages/types';
import { useTranslation } from '@/i18n';
import { formatMessageDate } from './utils';
import { Loading } from './Loading';

interface SystemMessageProps {
  message: IChatMessage;
  testId?: string;
}

// Currently we assume system message is used only for summary, in future that should change to utilize components prop or whatever ag-ui defines.
export const SystemMessage: React.FC<SystemMessageProps> = ({ message, testId }) => {
  const { t } = useTranslation();

  return (
    <div
      className="p-3 bg-card rounded flex-col justify-start items-start gap-3 flex mb-2.5 mr-2"
      data-testid={testId}
      key={message.id}
    >
      <div className="self-stretch justify-between items-center gap-1 inline-flex">
        <div className="grow shrink basis-0 h-6 justify-start items-center gap-2 flex">
          <div className="w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
            <Zap className="w-4 h-4 text-blue-600" />
          </div>
          <div className="text-sm font-semibold leading-tight">{t('Summarization')}</div>
          {message.created_at && (
            <div className="text-xs font-normal leading-[17px]">
              {formatMessageDate(message.created_at)}
            </div>
          )}
        </div>
      </div>
      <div className="self-stretch text-sm font-normal leading-tight">
        {message.in_progress ? (
          <Loading
            statusText={t("Condensing conversation history to stay within model's context limits")}
          />
        ) : message.error ? (
          <div className="text-destructive text-sm">{message.error}</div>
        ) : (
          <div>
            <div className="text-sm mb-2">
              {t(
                "Previous messages have been condensed to stay within model's context limits. Your full chat history is preserved."
              )}
            </div>
            <div className="whitespace-pre-wrap text-xs text-muted-foreground">
              {message.content}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
