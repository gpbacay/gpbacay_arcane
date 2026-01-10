/* Gilroy font placeholder - Replace with actual Gilroy font files */
/* Download Gilroy font and place the .woff2 files in this directory */
/* Required files:
   - Gilroy-Light.woff2
   - Gilroy-Regular.woff2
   - Gilroy-Medium.woff2
   - Gilroy-SemiBold.woff2
   - Gilroy-Bold.woff2
   - Gilroy-ExtraBold.woff2
*/

/* Fallback to system fonts if Gilroy is not available */
@font-face {
  font-family: 'Gilroy-Fallback';
  src: local('Inter'), local('system-ui'), local('-apple-system');
  font-weight: 100 900;
  font-display: swap;
}
