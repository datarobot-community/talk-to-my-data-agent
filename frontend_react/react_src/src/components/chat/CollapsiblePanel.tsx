import React, { useRef } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faChevronUp } from "@fortawesome/free-solid-svg-icons/faChevronUp";
import { faChevronDown } from "@fortawesome/free-solid-svg-icons/faChevronDown";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { useAppState } from "@/state";

interface CollapsiblePanelProps {
  header: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

export const CollapsiblePanel: React.FC<CollapsiblePanelProps> = ({
  header,
  children,
  defaultOpen,
}) => {
  const { collapsiblePanelDefaultOpen } = useAppState();
  const [isOpen, setIsOpen] = React.useState(
    defaultOpen !== undefined ? defaultOpen : collapsiblePanelDefaultOpen
  );
  const ref = useRef<HTMLDivElement>(null);
  
  //   useEffect(() => {
  //     ref.current?.scrollIntoView(false);
  //   });

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen} className="rounded border border-border">
      <CollapsibleTrigger asChild className="bg-muted">
        <div className="h-[52px] flex justify-between items-center px-4 cursor-pointer">
          <div>{header}</div>
          <FontAwesomeIcon icon={isOpen ? faChevronUp : faChevronDown} />
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent ref={ref} className="py-4 px-4 bg-muted">
        {children}
      </CollapsibleContent>
    </Collapsible>
  );
};
