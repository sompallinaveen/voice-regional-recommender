import { Roboto } from 'next/font/google';
import ThemeRegistry from '../ThemeRegistry';

const roboto = Roboto({
  weight: ['300', '400', '500', '700'],
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-roboto',
});

export const metadata = {
  title: 'SwaraRegional — Voice-Based Regional Accent Recommender',
  description:
    'Material Design frontend for acoustic regional accent recognition and localized recommendations.',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={roboto.variable}>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap"
        />
      </head>
      <body>
        <ThemeRegistry>{children}</ThemeRegistry>
      </body>
    </html>
  );
}
