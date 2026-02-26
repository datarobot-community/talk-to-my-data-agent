import React from 'react';
import {
  Collapsible,
  COLLAPSIBLE_VARIANT,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { CollapsibleChevron } from '@/components/ui/collapsible';
import { useAppState } from '@/state';

interface CollapsiblePanelProps {
  header: React.ReactNode;
  children: React.ReactNode;
  triggerTestId?: string;
}

export const CollapsiblePanel: React.FC<CollapsiblePanelProps> = ({
  header,
  children,
  triggerTestId = 'collapsible-panel-trigger',
}) => {
  const { collapsiblePanelDefaultOpen } = useAppState();
  const [isOpen, setIsOpen] = React.useState(collapsiblePanelDefaultOpen);

  return (
    <Collapsible
      open={isOpen}
      onOpenChange={setIsOpen}
      className="min-w-0 border border-border"
      variant={COLLAPSIBLE_VARIANT.standalone}
    >
      <CollapsibleTrigger data-testid={triggerTestId}>
        {header}
        <CollapsibleChevron />
      </CollapsibleTrigger>
      <CollapsibleContent className="min-w-0 overflow-hidden p-4">{children}</CollapsibleContent>
    </Collapsible>
  );
};
