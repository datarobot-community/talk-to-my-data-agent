import "./i18n";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter as Router } from "react-router-dom";
import "@fontsource/dm-sans/300.css";
import "@fontsource/dm-sans/400.css";
import "@fontsource/dm-sans/500.css";
import "@fontsource/dm-sans/600.css";
import "@fontsource/dm-sans/700.css";
import "./index.css";
import App from "./App.tsx";
import { AppStateProvider } from "./state";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

let basename = undefined;
if (import.meta.env.MODE === "production") {
  // eg. https://app.datarobot.com/custom_applications/{appId}/ -> /custom_applications/{appId}/
  basename = window.location.pathname.split("/").slice(0, 3).join("/");
}

const queryClient = new QueryClient();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <Router basename={basename}>
        <AppStateProvider>
          <App />
        </AppStateProvider>
      </Router>
    </QueryClientProvider>
  </StrictMode>
);
