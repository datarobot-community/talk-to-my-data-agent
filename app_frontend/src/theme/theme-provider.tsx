'use client';

import { useState, createContext, useContext, useLayoutEffect } from 'react';

type Theme = 'light' | 'dark';

interface ThemeContextType {
  theme: Theme;
  setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextType>({
  theme: 'light',
  setTheme: () => {},
});

export const useTheme = () => {
  return useContext(ThemeContext);
};

const themeKey = 'app-theme';

const getInitialTheme = (): Theme => {
  // TODO: remove this once we have a way to persist the theme
  return 'dark';
  // if (typeof window !== 'undefined') {
  //   const savedTheme = localStorage.getItem(themeKey);
  //   if (savedTheme) {
  //     return savedTheme as Theme;
  //   }
  //   return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  // }
  // return 'light';
};

export const ThemeProvider = ({
  children,
}: {
  children: React.ReactNode | ((props: { theme: Theme }) => React.ReactNode);
}) => {
  const [theme, setTheme] = useState<'light' | 'dark'>(getInitialTheme);

  useLayoutEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    localStorage.setItem(themeKey, theme);
  }, [theme]);

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {typeof children === 'function' ? children({ theme }) : children}
    </ThemeContext.Provider>
  );
};
