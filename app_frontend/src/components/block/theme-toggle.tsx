'use client';

import { Moon, Sun } from 'lucide-react';
import { useTheme } from '@/theme/theme-provider';
import { Button } from '@/components/ui/button';

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
    >
      <Sun
        className={`size-[1.2rem] scale-100 rotate-0 transition-all dark:scale-0 dark:-rotate-90`}
      />
      <Moon
        className={`absolute size-[1.2rem] scale-0 rotate-90 transition-all dark:scale-100 dark:rotate-0`}
      />
    </Button>
  );
}
