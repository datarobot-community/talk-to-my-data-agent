import { Button } from '@/components/ui/button';
import { Copy, Check } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';

interface CopyToClipboardButtonProps {
  content: string;
  label?: string;
  copiedLabel?: string;
  onCopy?: () => void;
}

export function CopyToClipboardButton({
  content,
  label = 'Copy to clipboard',
  copiedLabel = 'Copied!',
  onCopy,
}: CopyToClipboardButtonProps) {
  const [isCopied, setIsCopied] = useState(false);
  const timeoutRef = useRef<number | undefined>(undefined);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(content);
    setIsCopied(true);
    onCopy?.();

    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    timeoutRef.current = window.setTimeout(() => setIsCopied(false), 2000);
  };

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return (
    <Button variant="ghost" onClick={handleCopy}>
      {isCopied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
      {isCopied ? copiedLabel : label}
    </Button>
  );
}
