// API harness (libtest.html): nothing but the library's public exports on window.__pklib, so a driver
// (drive-lib.ts) can exercise them in any browser, and a button whose click runs window.__ontap (start() has to
// be called from a user gesture).
import * as lib from "../../lib/src/index";
const w = window as any;
w.__pklib = lib;
document.getElementById("tap")!.addEventListener("click", () => { w.__tapResult = w.__ontap?.(); });
