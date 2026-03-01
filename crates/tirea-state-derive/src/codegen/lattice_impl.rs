//! Code generation for `#[derive(Lattice)]`.
//!
//! Generates `impl Lattice for Foo` that delegates each named field to `Lattice::merge`.
//! Errors on tuple structs, unit structs, and enums.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields};

pub fn expand(input: &DeriveInput) -> syn::Result<TokenStream> {
    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(named) => &named.named,
            Fields::Unnamed(_) => {
                return Err(syn::Error::new_spanned(
                    input,
                    "#[derive(Lattice)] does not support tuple structs",
                ));
            }
            Fields::Unit => {
                return Err(syn::Error::new_spanned(
                    input,
                    "#[derive(Lattice)] does not support unit structs",
                ));
            }
        },
        Data::Enum(_) => {
            return Err(syn::Error::new_spanned(
                input,
                "#[derive(Lattice)] does not support enums",
            ));
        }
        Data::Union(_) => {
            return Err(syn::Error::new_spanned(
                input,
                "#[derive(Lattice)] does not support unions",
            ));
        }
    };

    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Build a where clause that adds `Lattice` bounds for each generic type parameter
    let extra_bounds: Vec<TokenStream> = input
        .generics
        .type_params()
        .map(|tp| {
            let ident = &tp.ident;
            quote! { #ident: tirea_state::Lattice }
        })
        .collect();

    let merged_where = if extra_bounds.is_empty() {
        quote! { #where_clause }
    } else if let Some(wc) = where_clause {
        let existing = &wc.predicates;
        quote! { where #existing, #(#extra_bounds),* }
    } else {
        quote! { where #(#extra_bounds),* }
    };

    let field_merges: Vec<TokenStream> = fields
        .iter()
        .map(|f| {
            let ident = f.ident.as_ref().unwrap();
            quote! {
                #ident: tirea_state::Lattice::merge(&self.#ident, &other.#ident)
            }
        })
        .collect();

    Ok(quote! {
        impl #impl_generics tirea_state::Lattice for #name #ty_generics #merged_where {
            fn merge(&self, other: &Self) -> Self {
                Self {
                    #(#field_merges,)*
                }
            }
        }
    })
}
