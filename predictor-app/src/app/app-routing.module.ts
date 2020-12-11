import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { MarketplaceComponent } from './marketplace/marketplace.component'
import { AboutUsComponent } from './about-us/about-us.component'
import { PredictionsComponent } from './predictions/predictions.component'

const routes: Routes = [
  { path: '', component: HomeComponent},
  { path: 'marketplace', component: MarketplaceComponent},
  { path: 'about-us', component: AboutUsComponent},
  { path: 'predictions', component: PredictionsComponent},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
