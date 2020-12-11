import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { ReactiveFormsModule, FormsModule } from '@angular/forms';
import { HttpClientModule } from "@angular/common/http";
import { MatButtonModule } from '@angular/material/button';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MatSelectModule } from '@angular/material/select';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatCardModule } from '@angular/material/card';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { HomeComponent } from './home/home.component';
import { PredictionsComponent } from './predictions/predictions.component';
import { MarketplaceComponent } from './marketplace/marketplace.component';
import { AboutUsComponent } from './about-us/about-us.component';
import { PredictionService } from './services/prediction.service';
import { MarketplaceService } from './services/marketplace.service';
import { SafePipe } from './safe.pipe';
import { ScriptHackComponent } from './script-hack/script-hack.component';

@NgModule({
  declarations: [
    AppComponent,
    HomeComponent,
    PredictionsComponent,
    MarketplaceComponent,
    AboutUsComponent,
    SafePipe,
    ScriptHackComponent
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,
    AppRoutingModule,
    ReactiveFormsModule,
    FormsModule,
    HttpClientModule,
    MatButtonModule,
    MatSelectModule,
    MatProgressSpinnerModule,
    MatCardModule
  ],
  providers: [PredictionService, MarketplaceService],
  bootstrap: [AppComponent]
})
export class AppModule { }
