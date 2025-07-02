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

export const WelcomeModal = () => {
  const { showWelcome, hideWelcomeModal } = useAppState();
  const [open, setOpen] = useState(showWelcome);

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
            Talk to my data
          </DialogTitle>
          <DialogDescription className="text-center mb-10">
            Use DataRobot’s intuitive chat-based analyst to ask questions about
            your data.
            <br />
            <br />
            Get started by selecting the datasets you want to work with.
          </DialogDescription>
        </DialogHeader>
        <Separator className="border-t mt-6" />
        <DialogFooter>
          <Button
            testId="welcome-modal-hide-button"
            onClick={() => {
              setOpen(false);
              hideWelcomeModal();
            }}
          >
            Select data
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
