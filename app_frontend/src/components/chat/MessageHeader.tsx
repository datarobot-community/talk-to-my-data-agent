import React, { JSX, useState } from 'react';
import { UserAvatar } from './Avatars';
import { Button } from '@/components/ui/button';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faTrash } from '@fortawesome/free-solid-svg-icons/faTrash';
import { faFileArrowDown } from '@fortawesome/free-solid-svg-icons/faFileArrowDown';
import { useTranslation } from '@/i18n';
import { ConfirmDialog } from '../ui-custom/confirm-dialog';

interface MessageHeaderProps {
  avatar?: () => JSX.Element;
  name: string;
  date: string;
  onDelete?: () => void;
  onExport?: () => void;
  isDeleting?: boolean;
  isResponseInProgress?: boolean;
  isResponseFailing?: boolean;
  isExporting?: boolean;
  chatId?: string;
  messageId?: string;
}

export const MessageHeader: React.FC<MessageHeaderProps> = ({
  avatar = UserAvatar,
  name,
  date,
  onDelete,
  onExport,
  isResponseInProgress = false,
  isResponseFailing = false,
  isDeleting = false,
  isExporting = false,
}) => {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);

  const isExportDisabled = isExporting || isResponseInProgress || isResponseFailing;
  const exportButtonTooltip = isResponseFailing
    ? t('Cannot export chat with errors')
    : isExporting
      ? t('Exporting...')
      : isResponseInProgress
        ? t('Wait for agent to finish responding')
        : t('Export chat');

  return (
    <>
      {onDelete && (
        <ConfirmDialog
          open={open}
          onOpenChange={setOpen}
          title={t('Delete message')}
          description={t('Are you sure you want to delete this message?')}
          onConfirm={onDelete}
          confirmText={t('Delete')}
          cancelText={t('Cancel')}
          variant="destructive"
          isLoading={isDeleting}
        />
      )}
      <div className="self-stretch justify-between items-center gap-1 inline-flex">
        <div className="grow shrink basis-0 h-6 justify-start items-center gap-2 flex">
          {avatar()}
          <div className="text-sm font-semibold leading-tight">{name}</div>
          <div className="text-xs font-normal leading-[17px]">{date}</div>
        </div>
        <div className="flex items-center">
          {onExport && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onExport()}
              title={exportButtonTooltip}
              disabled={isExportDisabled}
            >
              <FontAwesomeIcon icon={faFileArrowDown} />
            </Button>
          )}
          {onDelete && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setOpen(true)}
              title={t('Delete message and response')}
            >
              <FontAwesomeIcon icon={faTrash} />
            </Button>
          )}
        </div>
      </div>
    </>
  );
};
