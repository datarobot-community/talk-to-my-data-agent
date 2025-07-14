import React, { JSX, useState } from 'react';
import { UserAvatar } from './Avatars';
import { Button } from '@/components/ui/button';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faTrash } from '@fortawesome/free-solid-svg-icons/faTrash';
import { useTranslation } from '@/i18n';
import { ConfirmDialog } from '../ui-custom/confirm-dialog';
interface MessageHeaderProps {
  avatar?: () => JSX.Element;
  name: string;
  date: string;
  onDelete?: () => void;
  isLoading?: boolean;
}

export const MessageHeader: React.FC<MessageHeaderProps> = ({
  avatar = UserAvatar,
  name,
  date,
  onDelete,
  isLoading = false,
}) => {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);

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
          isLoading={isLoading}
        />
      )}
      <div className="self-stretch justify-between items-center gap-1 inline-flex">
        <div className="grow shrink basis-0 h-6 justify-start items-center gap-2 flex">
          {avatar()}
          <div className="text-sm font-semibold leading-tight">{name}</div>
          <div className="text-xs font-normal leading-[17px]">{date}</div>
        </div>
        {onDelete && (
          <Button
            variant="ghost"
            size="sm"
            className="h-6 w-6 p-0"
            onClick={e => {
              e.stopPropagation();
              setOpen(true);
            }}
            title={t('Delete message')}
          >
            <FontAwesomeIcon icon={faTrash} className="h-3.5 w-3.5" />
          </Button>
        )}
      </div>
    </>
  );
};
