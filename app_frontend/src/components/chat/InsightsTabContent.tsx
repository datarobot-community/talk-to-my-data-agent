import React from 'react';
import { HeaderSection } from './HeaderSection';
import { SuggestedQuestionsSection } from './SuggestedQuestionsSection';
import { MarkdownContent } from './MarkdownContent';
import { useTranslation } from '@/i18n';

interface InsightsTabContentProps {
  additionalInsights?: string | null;
  followUpQuestions?: string[] | null;
  chatId?: string;
  hasInProgressMessages: boolean;
}

export const InsightsTabContent: React.FC<InsightsTabContentProps> = ({
  additionalInsights,
  followUpQuestions,
  chatId,
  hasInProgressMessages,
}) => {
  const { t } = useTranslation();

  return (
    <>
      {additionalInsights && (
        <HeaderSection title={t('Data insights')}>
          <MarkdownContent content={additionalInsights} />
        </HeaderSection>
      )}
      <SuggestedQuestionsSection
        questions={followUpQuestions}
        chatId={chatId}
        hasInProgressMessages={hasInProgressMessages}
      />
    </>
  );
};
