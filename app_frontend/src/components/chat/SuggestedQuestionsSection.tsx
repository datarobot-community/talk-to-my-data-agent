import React from 'react';
import { SuggestedPrompt } from './SuggestedPrompt';

interface SuggestedQuestionsSectionProps {
  questions?: string[] | null;
  chatId?: string;
}

export const SuggestedQuestionsSection: React.FC<SuggestedQuestionsSectionProps> = ({
  questions,
  chatId,
}) => {
  if (!questions || questions.length === 0) {
    return null;
  }

  return (
    <>
      <div className="text-primary text-base font-semibold leading-tight">
        Suggested follow-up questions
      </div>
      <div className="mt-2 flex-col flex gap-2.5">
        {questions.map(q => (
          <SuggestedPrompt key={q} message={q} chatId={chatId} />
        ))}
      </div>
    </>
  );
};
