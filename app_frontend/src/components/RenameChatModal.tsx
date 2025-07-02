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
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPencil } from '@fortawesome/free-solid-svg-icons/faPencil';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useState } from 'react';
import { useRenameChat } from '@/api/chat-messages/hooks';

interface RenameChatModalProps {
    chatId: string;
    currentName: string;
}

export const RenameChatModal = ({ chatId, currentName }: RenameChatModalProps) => {
    const [name, setName] = useState(currentName);
    const [open, setOpen] = useState<boolean>(false);
    const { mutate: renameChat, isPending } = useRenameChat();

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
                <Button variant="link" className="ml-2">
                    <FontAwesomeIcon icon={faPencil} />
                </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[500px]">
                <DialogHeader>
                    <DialogTitle>Rename chat</DialogTitle>
                    <DialogDescription>Enter a new name for this chat.</DialogDescription>
                </DialogHeader>
                <div className="grid gap-4 py-4">
                    <div className="grid grid-cols-4 items-center gap-4">
                        <Label htmlFor="rename" className="text-right">
                            Chat name
                        </Label>
                        <Input
                            id="rename"
                            value={name}
                            onChange={event => setName(event.target.value)}
                            className="col-span-3"
                            placeholder="Enter a new name for this chat"
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
                        Cancel
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
                        {isPending ? 'Renaming...' : 'Rename'}
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
};
