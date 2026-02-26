import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Pencil } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useState } from 'react';
import { useRenameChat } from '@/api/chat-messages/hooks';
import { useTranslation } from '@/i18n';
import { MAX_CHAT_NAME_LENGTH } from '@/constants/chat';

interface RenameChatModalProps {
  chatId: string;
  currentName: string;
}

export const RenameChatModal = ({ chatId, currentName }: RenameChatModalProps) => {
  const [name, setName] = useState(currentName);
  const [open, setOpen] = useState<boolean>(false);
  const { mutate: renameChat, isPending } = useRenameChat();
  const { t } = useTranslation();
  return (
    <Dialog
      defaultOpen={false}
      open={open}
      onOpenChange={open => {
        if (open) {
          // Reset to current name when opening
          setName(currentName);
        }
        setOpen(open);
      }}
    >
      <DialogTrigger asChild>
        <Button variant="ghost" className="ml-2">
          <Pencil />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>{t('Rename chat')}</DialogTitle>
          <DialogDescription>{t('Enter a new name for this chat.')}</DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="rename" className="mn-label text-right">
              {t('Chat name')}
            </Label>
            <Input
              id="rename"
              value={name}
              onChange={event => setName(event.target.value)}
              maxLength={MAX_CHAT_NAME_LENGTH}
              className="col-span-3"
              placeholder={t('Enter a new name for this chat')}
              disabled={isPending}
              onKeyDown={event => {
                if (event.key === 'Enter' && name.trim() && name !== currentName) {
                  renameChat(
                    { chatId, name: name.trim() },
                    {
                      onSuccess: () => {
                        setOpen(false);
                      },
                    }
                  );
                }
              }}
            />
            {name.length >= MAX_CHAT_NAME_LENGTH && (
              <p className="col-span-3 col-start-2 text-xs text-destructive">
                {t('Chat name has reached maximum of {{max}} characters', {
                  max: MAX_CHAT_NAME_LENGTH,
                })}
              </p>
            )}
          </div>
        </div>
        <DialogFooter>
          <Button
            variant="ghost"
            onClick={() => {
              setName(currentName);
              setOpen(false);
            }}
          >
            {t('Cancel')}
          </Button>
          <Button
            onClick={() => {
              if (name.trim() && name !== currentName) {
                renameChat(
                  { chatId, name: name.trim() },
                  {
                    onSuccess: () => {
                      setOpen(false);
                    },
                  }
                );
              }
            }}
            disabled={isPending || !name.trim() || name === currentName}
          >
            {isPending ? t('Renaming...') : t('Rename')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
