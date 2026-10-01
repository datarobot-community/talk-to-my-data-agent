import { useState } from 'react';
import { ChevronDown, ChevronUp, Database, Plus, Upload } from 'lucide-react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { cn } from '@/lib/utils';
import { useTranslation } from '@/i18n';

import { AddDataModal } from './AddDataModal';
import { BrowseDataDrawer } from './BrowseDataDrawer';
import { ErrorBoundary } from './ErrorBoundary';

import './browse-data-styles.css';

/**
 * Single "Add Data" dropdown button. The menu lets the user pick how to add
 * data: "Browse Platform Data" opens the connectivity drawer (browse the data
 * registry and connect database tables), "Connect Database" opens the existing
 * upload/connect modal.
 */
export const AddDataMenu = ({ highlight }: { highlight?: boolean }) => {
  const { t } = useTranslation();
  const [menuOpen, setMenuOpen] = useState(false);
  const [addDataOpen, setAddDataOpen] = useState(false);
  const [browseOpen, setBrowseOpen] = useState(false);

  return (
    <>
      {/*
        modal={false}: a modal dropdown opening a dialog from a menu-item click
        leaves `pointer-events: none` stuck on <body> after the dialog closes
        (the page looks fine but is unclickable). Letting only the dialog manage
        the body lock avoids that. https://github.com/radix-ui/primitives/issues/1241
      */}
      <DropdownMenu open={menuOpen} onOpenChange={setMenuOpen} modal={false}>
        <DropdownMenuTrigger asChild>
          <Button
            variant="secondary"
            testId="add-data-button"
            data-highlight={highlight || undefined}
            className={cn(highlight && 'animate-(--animation-blink-border-and-shadow)', 'mr-2')}
          >
            <Plus /> {t('Add Data')}
            {menuOpen ? <ChevronUp /> : <ChevronDown />}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem data-testid="browse-data-menu-item" onClick={() => setBrowseOpen(true)}>
            <Database /> {t('Browse Platform Data')}
          </DropdownMenuItem>
          <DropdownMenuItem data-testid="add-data-menu-item" onClick={() => setAddDataOpen(true)}>
            <Upload /> {t('Connect Database')}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <AddDataModal open={addDataOpen} onOpenChange={setAddDataOpen} />
      {browseOpen && (
        // The drawer renders third-party connectivity/design-system components.
        // If one throws, close the drawer and tell the user rather than letting
        // the throw unmount the whole app (AECO-44).
        <ErrorBoundary
          label="browse-data-drawer"
          fallback={() => null}
          onError={() => {
            setBrowseOpen(false);
            toast.error(t('Browse Data could not be opened'));
          }}
        >
          <BrowseDataDrawer open={browseOpen} onOpenChange={setBrowseOpen} />
        </ErrorBoundary>
      )}
    </>
  );
};
