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
      className="border border-border min-w-0"
      variant={COLLAPSIBLE_VARIANT.standalone}
    >
      <CollapsibleTrigger data-testid={triggerTestId}>
        {header}
        <CollapsibleChevron />
      </CollapsibleTrigger>
      <CollapsibleContent className="py-4 px-4 min-w-0 overflow-hidden">
        {children}
      </CollapsibleContent>
    </Collapsible>
  );
};
