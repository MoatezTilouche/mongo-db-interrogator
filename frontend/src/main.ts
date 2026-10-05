import { provideZoneChangeDetection } from '@angular/core';
import { bootstrapApplication } from '@angular/platform-browser';
import { HttpBackend, HttpXhrBackend, provideHttpClient } from '@angular/common/http';
import { AppComponent } from './app/app.component';

bootstrapApplication(AppComponent, {
  providers: [
    provideZoneChangeDetection(),
    provideHttpClient(),
    // XHR exposes incremental SSE text through HttpClient download events.
    { provide: HttpBackend, useClass: HttpXhrBackend }
  ]
}).catch(() => {
  document.body.textContent = 'The application could not start. Please reload the page.';
});
