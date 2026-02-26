import { Button } from '@/components/ui/button';
import { useTranslation } from '@/i18n';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Plus } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useState } from 'react';
import { useCreateChat } from '@/api/chat-messages/hooks';
import { useNavigate } from 'react-router-dom';
import { generateChatRoute } from '@/pages/routes';
import { useAppState } from '@/state/hooks';
import { cn } from '@/lib/utils';
import { getChatName } from '@/api/chat-messages/utils';
import { MAX_CHAT_NAME_LENGTH } from '@/constants/chat';

type NewChatModalType = {
  highlight: boolean;
};

export const NewChatModal = ({ highlight }: NewChatModalType) => {
  const { t } = useTranslation();
  const [name, setName] = useState(() => getChatName());
  const [open, setOpen] = useState<boolean>(false);
  const { mutate: createChat, isPending } = useCreateChat();
  const navigate = useNavigate();
  const { dataSource } = useAppState();

  return (
    <Dialog
      defaultOpen={false}
      open={open}
      onOpenChange={open => {
        setName(getChatName());
        setOpen(open);
      }}
    >
      <DialogTrigger asChild>
        <Button
          className={cn(highlight && 'animate-(--animation-blink-border-and-shadow)', 'mr-2')}
          variant="secondary"
        >
          <Plus /> {t('New chat')}
        </Button>
      </DialogTrigger>
      <DialogContent
        className="sm:max-w-[500px]"
        // prevent focus from returning back to modal trigger, we want it to be autofocused on prompt-input
        onCloseAutoFocus={e => e.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle>{t('Create new chat')}</DialogTitle>
          <DialogDescription>
            {t('Creating a new chat does not affect any of your existing questions.')}
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="name" className="mn-label text-right">
              {t('Chat name')}
            </Label>
            <Input
              id="name"
              value={name}
              onChange={event => setName(event.target.value)}
              maxLength={MAX_CHAT_NAME_LENGTH}
              className="col-span-3"
              placeholder={t('Enter a name for your chat')}
              disabled={isPending}
              autoFocus
              autoComplete="off"
              onKeyDown={event => {
                if (event.key === 'Enter' && name.trim()) {
                  createChat(
                    { name: name.trim(), dataSource },
                    {
                      onSuccess: newChat => {
                        setOpen(false);
                        navigate(generateChatRoute(newChat.id));
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
              setName('');
              setOpen(false);
            }}
          >
            {t('Cancel')}
          </Button>
          <Button
            onClick={() => {
              if (name.trim()) {
                createChat(
                  { name: name.trim(), dataSource },
                  {
                    onSuccess: newChat => {
                      setOpen(false);
                      navigate(generateChatRoute(newChat.id));
                    },
                  }
                );
              }
            }}
            disabled={isPending || !name.trim()}
          >
            {isPending ? t('Creating...') : t('Create chat')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
