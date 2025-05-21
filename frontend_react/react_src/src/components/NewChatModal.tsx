import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faPlus } from "@fortawesome/free-solid-svg-icons/faPlus";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useState } from "react";
import { useCreateChat } from "@/api-state/chat-messages/hooks";
import { useNavigate } from "react-router-dom";
import { generateChatRoute } from "@/pages/routes";
import { useAppState } from "@/state/hooks";
import { useTranslation } from "react-i18next";

export const NewChatModal = () => {
  const [name, setName] = useState("");
  const [open, setOpen] = useState<boolean>(false);
  const { mutate: createChat, isPending } = useCreateChat();
  const navigate = useNavigate();
  const { dataSource } = useAppState();
  const { t } = useTranslation();
  return (
    <Dialog
      defaultOpen={false}
      open={open}
      onOpenChange={(open) => {
        setName("");
        setOpen(open);
      }}
    >
      <DialogTrigger asChild>
        <Button variant="outline">
          <FontAwesomeIcon icon={faPlus} /> {t("new_chat")}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>{t("create_new_chat")}</DialogTitle>
          <DialogDescription>
            {t("create_chat_description")}
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="name" className="text-right">
              {t("chat_name")}
            </Label>
            <Input
              id="name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              className="col-span-3"
              placeholder={t("enter_chat_name")}
              disabled={isPending}
              onKeyDown={(event) => {
                if (event.key === "Enter" && name.trim()) {
                  createChat(
                    { name: name.trim(), dataSource },
                    {
                      onSuccess: (newChat) => {
                        setOpen(false);
                        navigate(generateChatRoute(newChat.id));
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
              setName("");
              setOpen(false);
            }}
          >
            {t("cancel")}
          </Button>
          <Button
            onClick={() => {
              if (name.trim()) {
                createChat(
                  { name: name.trim(), dataSource },
                  {
                    onSuccess: (newChat) => {
                      setOpen(false);
                      navigate(generateChatRoute(newChat.id));
                    },
                  }
                );
              }
            }}
            disabled={isPending || !name.trim()}
          >
            {isPending ? t("creating_new_chat") : t("create_chat")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
