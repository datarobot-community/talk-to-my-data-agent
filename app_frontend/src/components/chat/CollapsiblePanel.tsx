import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faChevronUp } from '@fortawesome/free-solid-svg-icons/faChevronUp';
import { faChevronDown } from '@fortawesome/free-solid-svg-icons/faChevronDown';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Button } from '@/components/ui/button';
import { useAppState } from '@/state';
import { useTranslation } from '@/i18n';

interface CollapsiblePanelProps {
  header: React.ReactNode;
  children: React.ReactNode;
  headerActions?: React.ReactNode;
  triggerTestId?: string;
}

export const CollapsiblePanel: React.FC<CollapsiblePanelProps> = ({
  header,
  children,
  headerActions,
  triggerTestId = 'collapsible-panel-trigger',
}) => {
  const { t } = useTranslation();
  const { collapsiblePanelDefaultOpen } = useAppState();
  const [isOpen, setIsOpen] = React.useState(collapsiblePanelDefaultOpen);

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen} className="rounded border border-border">
      <CollapsibleTrigger asChild className="bg-muted">
        <div className="h-[52px] flex justify-between items-center px-4 cursor-pointer">
          <div>{header}</div>
          <div className="flex items-center gap-2">
            {headerActions}
            <Button
              variant="ghost"
              size="icon"
              testId={triggerTestId}
              aria-label={t('Toggle panel')}
              aria-expanded={isOpen}
            >
              <FontAwesomeIcon icon={isOpen ? faChevronUp : faChevronDown} />
            </Button>
          </div>
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent className="py-4 px-4 bg-muted">{children}</CollapsibleContent>
    </Collapsible>
  );
};
