import { Button } from "@/components/ui/button";
import playgroundMidnight from "@/assets/playground-midnight.svg";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useAppState } from "@/state";
import { useState } from "react";
import { Separator } from "./ui/separator";
import { useTranslation } from "react-i18next";

export const WelcomeModal = () => {
  const { showWelcome, hideWelcomeModal } = useAppState();
  const [open, setOpen] = useState(showWelcome);
  const { t } = useTranslation();

  return (
    <Dialog
      defaultOpen={showWelcome}
      open={open}
      onOpenChange={(open) => !open && setOpen(open)}
    >
      <DialogContent className="sm:max-w-[725px]">
        <div className="grid gap-4 py-4">
          <div className="grid justify-center gap-4">
            <img src={playgroundMidnight} alt="" />
          </div>
        </div>
        <DialogHeader>
          <DialogTitle className="text-center mb-4">
            {t("talk_to_my_data")}
          </DialogTitle>
          <DialogDescription className="text-center mb-10">
            {t("welcome_description")}
            <br />
            <br />
            {t("welcome_get_started")}
          </DialogDescription>
        </DialogHeader>
        <Separator className="border-t mt-6" />
        <DialogFooter>
          <Button
            onClick={() => {
              setOpen(false);
              hideWelcomeModal();
            }}
          >
            {t("select_data")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
