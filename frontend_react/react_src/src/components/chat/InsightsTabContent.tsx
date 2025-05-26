import React from "react";
import { HeaderSection } from "./HeaderSection";
import { SuggestedQuestionsSection } from "./SuggestedQuestionsSection";
import { MarkdownContent } from "./MarkdownContent";
import { CollapsiblePanel } from "./CollapsiblePanel";
import { useAppState } from "@/state";

interface InsightsTabContentProps {
  additionalInsights?: string | null;
  followUpQuestions?: string[] | null;
  chatId?: string;
}

export const InsightsTabContent: React.FC<InsightsTabContentProps> = ({
  additionalInsights,
  followUpQuestions,
  chatId,
}) => {
  const { expandGraphsInsightsDefaultOpen } = useAppState();

  return (
    <CollapsiblePanel header="Insights" defaultOpen={expandGraphsInsightsDefaultOpen}>
      {/* <InfoText>
        DataRobot generates additional content based on your original question.
      </InfoText> */}
      {additionalInsights && (
        <HeaderSection title="Data insights">
          <MarkdownContent content={additionalInsights} />
        </HeaderSection>
      )}
      <SuggestedQuestionsSection questions={followUpQuestions} chatId={chatId}/>
    </CollapsiblePanel>
  );
};
